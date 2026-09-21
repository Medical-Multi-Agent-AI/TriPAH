import argparse
import json
import time
import csv
import os
import re
from pathlib import Path

import pandas as pd
from tqdm import tqdm


LABELS_14 = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Enlarged Cardiomediastinum",
    "Fracture",
    "Lung Lesion",
    "Lung Opacity",
    "Pleural Effusion",
    "Pneumonia",
    "Pneumothorax",
    "Pleural Other",
    "Support Devices",
    "No Finding",
]


PROMPT_INSTRUCTIONS = (
    "You are a radiology labeling assistant for chest X-ray reports.\n"
    "Task: Read the report text (impression preferred, otherwise findings) and map to the following 14 labels.\n"
    "You are also provided MeSH terms and Problems keywords extracted automatically; treat them as weak signals to assist mapping.\n"
    "If MeSH/Problems suggest a finding but the report text is unclear or contradictory, set that label to 0.5 (uncertain).\n"
    "Output a strict JSON object with EXACT keys (Atelectasis, Cardiomegaly, Consolidation, Edema, Enlarged Cardiomediastinum, Fracture, Lung Lesion, Lung Opacity, Pleural Effusion, Pneumonia, Pneumothorax, Pleural Other, Support Devices, No Finding).\n"
    "Values must be one of: 0, 0.5, 1, where: 1=clearly present; 0=clearly absent; 0.5=uncertain/suggestive/contradictory or low-confidence.\n"
    "Rules: If any positive finding is present (value=1) then 'No Finding' MUST be 0. Only set 'No Finding'=1 when no abnormal findings are present.\n"
    "Do not hallucinate or infer unsupported findings.\n"
    "Respond ONLY with the JSON object and nothing else.\n"
)


def _build_message(report_text: str, mesh: str, problems: str) -> list[dict]:
    report_text = str(report_text or "")
    report_text = " ".join(report_text.split())[:4096]
    mesh_text = " ".join(str(mesh or "").split())[:1024]
    prob_text = " ".join(str(problems or "").split())[:1024]
    content = (
        f"{PROMPT_INSTRUCTIONS}\n"
        f"Report (Impression/Findings):\n{report_text}\n\n"
        f"MeSH terms (weak signals):\n{mesh_text}\n"
        f"Problems keywords (weak signals):\n{prob_text}\n"
    )
    return [{"role": "user", "content": content}]


def _normalize_labels(d: dict) -> dict:
    # ensure all keys exist and values in {0, 0.5, 1}
    out = {}
    for k in LABELS_14:
        v = d.get(k, 0)
        try:
            v = float(v)
        except Exception:
            v = 0.0
        # clamp to allowed set
        if v not in (0.0, 0.5, 1.0):
            # nearest allowed
            v = min((0.0, 0.5, 1.0), key=lambda t: abs(t - v))
        out[k] = v
    # enforce No Finding rule strictly:
    # Only set No Finding = 1 when all other 13 labels are exactly 0;
    # otherwise set No Finding = 0 (including any 0.5 or 1 present).
    others = [out[k] for k in LABELS_14 if k != "No Finding"]
    if all(v == 0.0 for v in others):
        out["No Finding"] = 1.0
    else:
        out["No Finding"] = 0.0
    return out


def run(args):
    if not args.api_key:
        raise RuntimeError("Missing API key. Set XF_API_KEY or pass --api_key.")

    src = Path(args.csv)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(src)
    # expected columns: uid, impression, findings, MeSH, Problems
    if "uid" not in df.columns:
        raise ValueError("indiana_reports.csv must include 'uid' column")
    # pick text: impression preferred else findings else empty; also collect MeSH/Problems
    rows_src = []
    for _, r in df.iterrows():
        t = r.get("impression")
        if not isinstance(t, str) or not t.strip():
            t = r.get("findings")
        text = t if isinstance(t, str) else ""
        mesh = r.get("MeSH") if "MeSH" in df.columns else ""
        problems = r.get("Problems") if "Problems" in df.columns else ""
        rows_src.append((r.get("uid"), text, mesh, problems))

    # setup client (XFyun Spark via OpenAI SDK base_url)
    try:
        from openai import OpenAI
    except Exception as e:
        raise RuntimeError("Please 'pip install openai' to use the HTTP client") from e

    from openai import OpenAI
    client = OpenAI(api_key=args.api_key, base_url=args.api_base)
    alt_base = args.api_base.replace("https://", "http://") if args.api_base.startswith("https://") else args.api_base.replace("http://", "https://")

    # prepare CSV writer (incremental writes)
    fieldnames = ["uid"] + LABELS_14 + ["mesh_raw", "problems_raw"]
    n_total = len(rows_src)
    n_written = 0
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, (uid, text, mesh, problems) in enumerate(tqdm(rows_src, total=n_total, desc="LLM labeling", unit="sample")):
            # skip empty fully
            if not str(text).strip():
                labels = {k: 0.0 for k in LABELS_14}
                labels["No Finding"] = 1.0
                source = "empty"
            else:
                error_msg = None
                # try up to 2 attempts: primary base_url then alternate (switch http/https)
                # decide JSON mode support: only DeepSeek R1/V3 or explicit override
                model_lower = str(args.model_id or "").lower()
                use_json_mode = (
                    getattr(args, "json_mode", "auto") == "on" or (
                        getattr(args, "json_mode", "auto") == "auto" and ("deepseek" in model_lower or "v3" in model_lower or "r1" in model_lower)
                    )
                )

                for attempt in range(2):
                    try:
                        this_client = client if attempt == 0 else OpenAI(api_key=args.api_key, base_url=alt_base)
                        extra_body = {
                            "search_disable": True,
                        }
                        if use_json_mode:
                            extra_body["response_format"] = {"type": "json_object"}

                        resp = this_client.chat.completions.create(
                            model=args.model_id,
                            messages=_build_message(text, mesh, problems),
                            temperature=0.0,
                            max_tokens=1024,
                            extra_headers={"lora_id": os.environ.get("XF_LORA_ID", "0")},
                            stream_options={"include_usage": True},
                            extra_body=extra_body,
                        )
                        msg = resp.choices[0].message
                        content = msg.content or "{}"
                        # Try strict JSON parse, else naive brace extraction
                        parsed = None
                        try:
                            parsed = json.loads(content)
                        except Exception:
                            m = re.search(r"\{[\s\S]*\}", content)
                            if m:
                                try:
                                    parsed = json.loads(m.group(0))
                                except Exception:
                                    parsed = None
                        if parsed is None:
                            raise ValueError("Non-JSON response")
                        labels = _normalize_labels(parsed)
                        source = "llm" if attempt == 0 else "llm(alt_base)"
                        error_msg = None
                        break
                    except Exception as e:
                        error_msg = str(e)
                        # brief backoff before next attempt
                        time.sleep(min(1.0, args.sleep))
                if error_msg is not None:
                    # fallback: mark unknowns as 0 and No Finding=0.5
                    labels = {k: 0.0 for k in LABELS_14}
                    labels["No Finding"] = 0.5
                    source = f"fallback:{error_msg.split(':')[0]}"
                # polite rate limiting
                time.sleep(args.sleep)

            # print per-sample normalized labels
            print(f"[{i+1}/{n_total}] uid={uid} source={source} labels={json.dumps(labels, ensure_ascii=False)}", flush=True)

            # write per-sample to CSV
            row = {"uid": uid, "mesh_raw": str(mesh or ""), "problems_raw": str(problems or "")}
            row.update(labels)
            writer.writerow(row)
            n_written += 1
            f.flush()

    print(f"✓ Saved {out} with {n_written} rows and {len(LABELS_14)} labels per row")


def main():
    from utils.image_path import iuxray_root
    parser = argparse.ArgumentParser(description="Extract IU-Xray labels via XFyun LLM JSON Mode")
    parser.add_argument("--csv", type=str, default=str(iuxray_root / "indiana_reports.csv"))
    parser.add_argument("--out", type=str, default=str(iuxray_root / "labels_llm.csv"))
    parser.add_argument("--api_key", type=str, default=os.environ.get("XF_API_KEY"), help="XFyun API key; defaults to XF_API_KEY")
    parser.add_argument("--api_base", type=str, default="https://maas-api.cn-huabei-1.xf-yun.com/v1")
    parser.add_argument("--model_id", type=str, default="xop3qwen1b7", help="XFyun modelId")
    parser.add_argument("--sleep", type=float, default=0.1, help="sleep seconds between requests to avoid rate limit")
    parser.add_argument("--json_mode", type=str, default="auto", choices=["auto","on","off"], help="JSON Mode: auto enables only for DeepSeek R1/V3, on forces, off disables")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
