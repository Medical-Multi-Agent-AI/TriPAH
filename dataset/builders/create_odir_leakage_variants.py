#!/usr/bin/env python3
"""Create ODIR caption/prompt variants for leakage auditing.

The original converted ODIR caption text contains diagnostic keywords. This
utility builds less diagnostic variants while keeping the sample order fixed, so
the same index/label/image files can be reused.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

from model.open_clip.tokenizer import SimpleTokenizer


PATTERN = re.compile(
    r"The Patient Age is (?P<age>.*?), "
    r"The Patient Sex is (?P<sex>.*?), "
    r"The Patient Left-Diagnostic Keywords is (?P<left>.*?), "
    r"The Patient Right-Diagnostic Keywords is (?P<right>.*?)[.]?$"
)

NORMAL_HINTS = (
    "normal",
    "normal fundus",
    "no apparent abnormality",
    "no obvious abnormality",
)


def _parse_caption(line: str) -> dict[str, str]:
    match = PATTERN.match(line.strip())
    if not match:
        return {
            "age": "unknown",
            "sex": "patient",
            "left": "",
            "right": "",
        }
    return {k: v.strip() for k, v in match.groupdict().items()}


def _is_normal(text: str) -> bool:
    lowered = text.lower()
    return any(hint in lowered for hint in NORMAL_HINTS)


def _finding_text(keyword: str) -> str:
    return "normal finding" if _is_normal(keyword) else "abnormal finding"


def make_variant(line: str, mode: str) -> str:
    item = _parse_caption(line)
    age = item["age"]
    sex = item["sex"]
    left = item["left"]
    right = item["right"]

    if mode == "masked":
        return (
            f"The Patient Age is {age}, The Patient Sex is {sex}, "
            f"The Patient Left Fundus Finding is {_finding_text(left)}, "
            f"The Patient Right Fundus Finding is {_finding_text(right)}."
        )
    if mode == "meta_only":
        return (
            f"The Patient Age is {age}, The Patient Sex is {sex}. "
            "The patient has bilateral fundus images."
        )
    if mode == "generic":
        return "A bilateral fundus image."
    raise ValueError(f"Unsupported mode: {mode}")


def write_caption(path: Path, captions: list[str]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for caption in captions:
            f.write(caption.strip() + "\n")


def write_prompt_ids(path: Path, captions: list[str], context_length: int) -> None:
    tokenizer = SimpleTokenizer()
    ids = []
    for caption in captions:
        token_ids = tokenizer([caption], context_length=context_length)[0]
        ids.append(token_ids.cpu().numpy().astype(np.int64))
    np.savez_compressed(path, prompt_caption_ids=np.stack(ids, axis=0))


def main() -> None:
    parser = argparse.ArgumentParser(description="Create ODIR leakage-audit caption and prompt variants.")
    parser.add_argument("--dataset-dir", default="dataset/odir", help="Converted ODIR dataset directory.")
    parser.add_argument("--context-length", type=int, default=77, help="CLIP token context length.")
    parser.add_argument(
        "--modes",
        default="masked,meta_only,generic",
        help="Comma-separated variants: masked,meta_only,generic.",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    source = dataset_dir / "caption.txt"
    if not source.exists():
        raise FileNotFoundError(f"Missing source captions: {source}")

    source_captions = [line.strip() for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]

    for mode in modes:
        captions = [make_variant(line, mode) for line in source_captions]
        caption_path = dataset_dir / f"caption_{mode}.txt"
        prompt_path = dataset_dir / f"prompt_caption_{mode}.npz"
        write_caption(caption_path, captions)
        write_prompt_ids(prompt_path, captions, args.context_length)
        print(f"[OK] {mode}: {caption_path} | {prompt_path} | {len(captions)} samples")


if __name__ == "__main__":
    main()
