import os
import json
import sys
from pathlib import Path
from typing import List, Tuple, Dict
import math

import numpy as np
import pandas as pd
from PIL import Image
import scipy.io as sio
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


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


# Minimal synonym map for MeSH/Problems to our 14 labels
SYNONYM_MAP = {
    # direct terms
    "atelectasis": "Atelectasis",
    "cardiomegaly": "Cardiomegaly",
    "consolidation": "Consolidation",
    "edema": "Edema",
    "enlarged cardiomediastinum": "Enlarged Cardiomediastinum",
    "fracture": "Fracture",
    "lesion": "Lung Lesion",
    "lung lesion": "Lung Lesion",
    "opacity": "Lung Opacity",
    "lung opacity": "Lung Opacity",
    "pleural effusion": "Pleural Effusion",
    "pneumonia": "Pneumonia",
    "pneumothorax": "Pneumothorax",
    "pleural calcification": "Pleural Other",
    "pleural thickening": "Pleural Other",
    "support devices": "Support Devices",
    "device": "Support Devices",
    "catheter": "Support Devices",
    "line": "Support Devices",
    "no finding": "No Finding",
}


NEGATIVE_NORMAL_PATTERNS = [
    "no acute cardiopulmonary process",
    "no acute cardiopulmonary disease",
    "no acute cardiopulmonary abnormality",
    "no acute process",
    "no acute disease",
    "no focal airspace consolidation",
    "no pneumothorax",
    "no pleural effusion",
    "no abnormality",
    "normal chest",
    "normal study",
]


def _clean_text(text, max_chars: int = 512) -> str:
    """Robust text cleaner that treats None/NaN/"nan" as empty.

    - Handles pandas/numpy NaN values
    - Safely decodes bytes
    - Normalizes whitespace and trims
    """
    # None or NA-like -> empty
    if text is None:
        return ""
    try:
        # pandas-aware NA check
        if pd.isna(text):
            return ""
    except Exception:
        pass

    # bytes -> str
    if isinstance(text, (bytes, bytearray)):
        try:
            text = text.decode('utf-8', errors='ignore')
        except Exception:
            text = str(text)

    # float NaN -> empty
    try:
        if isinstance(text, (float, np.floating)) and math.isnan(float(text)):
            return ""
    except Exception:
        pass

    s = str(text).strip()
    # common nan-like strings -> empty
    if s.lower() in {"nan", "none", "null"}:
        return ""
    s = s.replace("___", "[BLANK]")
    s = " ".join(s.split())
    return s[:max_chars]


def _split_terms(s: str) -> List[str]:
    if not s:
        return []
    # split on semicolon or comma
    parts = []
    for token in s.replace("|", ",").split(";"):
        parts.extend(token.split(","))
    return [p.strip().lower() for p in parts if p.strip()]


def _map_labels(mesh: str, problems: str, report_text: str) -> np.ndarray:
    terms = set(_split_terms(mesh)) | set(_split_terms(problems))
    vec = np.zeros((len(LABELS_14),), dtype=np.float32)

    # direct mapping
    for t in terms:
        if t in SYNONYM_MAP:
            label = SYNONYM_MAP[t]
            idx = LABELS_14.index(label)
            vec[idx] = 1.0

    # lightweight rule back-off only when no positives yet
    if vec.sum() == 0:
        # Robustly handle NaN/non-string report_text
        if report_text is None:
            rt = ""
        elif isinstance(report_text, float):
            try:
                import math
                rt = "" if math.isnan(report_text) else str(report_text)
            except Exception:
                rt = ""
        else:
            rt = str(report_text)
        rt = rt.lower()
        # strong normals -> No Finding
        if any(p in rt for p in NEGATIVE_NORMAL_PATTERNS):
            vec[LABELS_14.index("No Finding")] = 1.0
        # a few precise positive hints
        for key, lbl in [
            ("pleural effusion", "Pleural Effusion"),
            ("pneumothorax", "Pneumothorax"),
            ("consolidation", "Consolidation"),
            ("atelectasis", "Atelectasis"),
            ("cardiomegaly", "Cardiomegaly"),
            ("opacity", "Lung Opacity"),
        ]:
            if key in rt:
                vec[LABELS_14.index(lbl)] = 1.0

    return vec


def _load_embeddings_from_indication(indications: List[str]) -> np.ndarray:
    try:
        import torch
        from model import open_clip
        device = "cuda" if torch.cuda.is_available() else "cpu"
        clip_arch = 'ViT-B-16-quickgelu'
        clip_model, _, _ = open_clip.create_model_and_transforms(clip_arch, pretrained='metaclip_fullcc')
        clip_model = clip_model.to(device).eval()
        tokenizer = open_clip.get_tokenizer(clip_arch)

        bs = 512
        feats = []
        with torch.no_grad():
            for i in range(0, len(indications), bs):
                caps = [ _clean_text(s or "", 512) for s in indications[i:i+bs] ]
                tok = tokenizer(caps).to(device)  # [B, 77]
                if hasattr(clip_model, 'text'):
                    x = clip_model.text.token_embedding(tok)
                    x = x + clip_model.text.positional_embedding
                else:
                    x = clip_model.token_embedding(tok)
                    x = x + clip_model.positional_embedding
                x_np = x.detach().cpu().numpy().astype('float32')
                # Ensure [B, 77, 512] by explicit truncate/pad
                if x_np.shape[1] > 77:
                    x_np = x_np[:, :77, :]
                elif x_np.shape[1] < 77:
                    pad = np.zeros((x_np.shape[0], 77 - x_np.shape[1], x_np.shape[2]), dtype='float32')
                    x_np = np.concatenate([x_np, pad], axis=1)
                feats.append(x_np)
        return np.concatenate(feats, axis=0)
    except Exception:
        N = len(indications)
        return np.zeros((N, 77, 512), dtype=np.float32)


def _horizontal_stitch(img_l: Image.Image, img_r: Image.Image, target_short: int = 384) -> Image.Image:
    # unify height to target_short, keep aspect ratio
    h_l = target_short
    w_l = int(img_l.width * (h_l / img_l.height))
    img_l_rs = img_l.resize((w_l, h_l), Image.BICUBIC)

    h_r = target_short
    w_r = int(img_r.width * (h_r / img_r.height))
    img_r_rs = img_r.resize((w_r, h_r), Image.BICUBIC)

    stitched = Image.new('RGB', (w_l + w_r, target_short))
    stitched.paste(img_l_rs, (0, 0))
    stitched.paste(img_r_rs, (w_l, 0))
    return stitched


class IUXrayConverter:
    def __init__(self, iu_root: Path, output_root: Path, image_resolution: int = 384, n_workers: int | None = None, embed_batch_size: int = 256):
        self.iu_root = Path(iu_root)
        self.output_root = Path(output_root)
        self.image_dir = self.iu_root / 'images'
        self.reports_csv = self.iu_root / 'indiana_reports.csv'
        self.proj_csv = self.iu_root / 'indiana_projections.csv'
        self.image_resolution = image_resolution
        self.n_workers = n_workers or max(1, min(32, (os.cpu_count() or 8)))
        self.embed_batch_size = embed_batch_size

        self.output_images = self.output_root / 'images'
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.output_images.mkdir(parents=True, exist_ok=True)

        self.df = None
        self.rows = []  # list of dicts with uid, caption, prompt, labels, index_path
        self.llm_labels: dict[str, np.ndarray] = {}

    def load_merge(self):
        rep = pd.read_csv(self.reports_csv)
        proj = pd.read_csv(self.proj_csv)
        # ensure expected columns exist
        needed_rep = {"uid", "MeSH", "Problems", "image", "indication", "findings", "impression"}
        needed_proj = {"uid", "filename", "projection"}
        if not needed_rep.issubset(rep.columns):
            missing = needed_rep - set(rep.columns)
            raise ValueError(f"Missing report columns: {missing}")
        if not needed_proj.issubset(proj.columns):
            missing = needed_proj - set(proj.columns)
            raise ValueError(f"Missing projection columns: {missing}")

        # group filenames by uid
        g = proj.groupby('uid')
        # build dictionary uid -> dict(frontal=filename, lateral=filename)
        pairs = {}
        for uid, dfu in g:
            frontal = None
            lateral = None
            # prioritize PA/AP as frontal
            for _, r in dfu.iterrows():
                pj = str(r['projection']).upper()
                fn = str(r['filename'])
                if frontal is None and (pj == 'PA' or pj == 'AP' or 'FRONTAL' in pj):
                    frontal = fn
                if lateral is None and 'LATERAL' in pj:
                    lateral = fn
            # fallback any image if missing
            if frontal is None and len(dfu) > 0:
                frontal = str(dfu.iloc[0]['filename'])
            pairs[uid] = {"frontal": frontal, "lateral": lateral}

        # join with reports
        self.df = rep.copy()
        self.df['frontal'] = self.df['uid'].map(lambda u: pairs.get(u, {}).get('frontal'))
        self.df['lateral'] = self.df['uid'].map(lambda u: pairs.get(u, {}).get('lateral'))

        # optionally load LLM labels if available (preserve 0.5 uncertainty)
        llm_path = self.iu_root / 'labels_llm.csv'
        if llm_path.exists():
            try:
                llm_df = pd.read_csv(llm_path)
                if 'uid' in llm_df.columns and all(lbl in llm_df.columns for lbl in LABELS_14):
                    # coerce to numeric and fill NaN with 0.0
                    for lbl in LABELS_14:
                        llm_df[lbl] = pd.to_numeric(llm_df[lbl], errors='coerce').fillna(0.0)
                    # build uid -> np.ndarray[14] float32
                    self.llm_labels = {
                        str(r['uid']): np.array([r[lbl] for lbl in LABELS_14], dtype=np.float32)
                        for _, r in llm_df.iterrows()
                    }
                    print(f"✓ Loaded LLM labels: {len(self.llm_labels)} rows from {llm_path}")
                else:
                    print(f"! labels_llm.csv missing required columns; falling back to rule-based labels")
            except Exception as e:
                print(f"! Failed to read LLM labels ({e}); falling back to rule-based labels")

    def _make_image(self, uid: str, frontal: str, lateral: str) -> str:
        # returns relative path like f"{uid}.jpg"
        out_name = f"{uid}.jpg"
        out_path = self.output_images / out_name

        def _open(fn: str) -> Image.Image:
            img = Image.open(self.image_dir / fn).convert('RGB')
            return img

        try:
            if frontal and lateral and frontal != lateral:
                img_l = _open(frontal)
                img_r = _open(lateral)
                stitched = _horizontal_stitch(img_l, img_r, target_short=self.image_resolution)
                stitched.save(out_path, format='JPEG', quality=92)
            else:
                img = _open(frontal or lateral)
                # resize shortest side
                h = self.image_resolution
                w = int(img.width * (h / img.height))
                img = img.resize((w, h), Image.BICUBIC)
                img.save(out_path, format='JPEG', quality=92)
        except Exception:
            # if any failure, skip creating image by returning empty path
            return ""
        return out_name

    def _process_row(self, r: dict) -> dict | None:
        try:
            uid = str(r.get('uid'))
            frontal = r.get('frontal')
            lateral = r.get('lateral')
            rel = self._make_image(uid, frontal, lateral)
            if not rel:
                return None

            cap_raw = r.get('impression') or r.get('findings') or ""
            caption = _clean_text(cap_raw, 512)
            if not caption:
                return None

            prompt = _clean_text(r.get('indication') or "", 512)
            # prefer LLM-derived labels if available (keeps 0.5 values), else fallback to rule-based
            llm_vec = self.llm_labels.get(uid)
            if llm_vec is not None:
                labels = llm_vec.astype(np.float32, copy=False)
            else:
                labels = _map_labels(str(r.get('MeSH') or ""), str(r.get('Problems') or ""), (r.get('impression') or r.get('findings') or ""))

            return {
                "uid": uid,
                "caption": caption,
                "prompt": prompt,
                "labels": labels,
                "index": f"images/{rel}",
            }
        except Exception:
            return None

    def build_rows(self):
        records = self.df.to_dict(orient='records')
        rows: list[dict] = []
        with ThreadPoolExecutor(max_workers=self.n_workers) as ex:
            for result in tqdm(ex.map(self._process_row, records), total=len(records), desc="Building rows", unit="sample"):
                if result is not None:
                    rows.append(result)
        self.rows = rows

    def write_outputs(self):
        if not self.rows:
            raise RuntimeError("No rows built; nothing to write.")

        # caption.txt
        cap_path = self.output_root / 'caption.txt'
        with open(cap_path, 'w', encoding='utf-8') as f:
            for row in self.rows:
                f.write(row['caption'] + "\n")

        # index.mat (save as Nx1 unicode array to avoid 1xN row-vector issue)
        index_path = self.output_root / 'index.mat'
        paths = [row['index'] for row in self.rows]
        max_len = max((len(p) for p in paths), default=1)
        index_arr = np.array(paths, dtype=f"<U{max_len}").reshape(-1, 1)
        sio.savemat(index_path, {"index": index_arr}, do_compression=True)

        # label.mat
        label_path = self.output_root / 'label.mat'
        labels_mat = np.stack([row['labels'] for row in self.rows], axis=0).astype(np.float32, copy=False)
        sio.savemat(
            label_path,
            {
                "labels": labels_mat,
                "category": labels_mat,  # compatibility with older loaders expecting 'category' as matrix
                "label_names": np.array(LABELS_14, dtype=object),
            },
            do_compression=True,
        )

        # prompt_caption.npz: 保存两种形式
        # 1) prompt_caption_ids: tokenizer 的 token ids，形状 [N,77]
        # 2) prompt_caption: 通过 CLIP token+positional+transformer 编码后的嵌入，形状 [N,77,512]
        prompts = [row['prompt'] for row in self.rows]
        prompt_ids, prompt_feats = _encode_prompts_ids_and_embeddings_with_progress(prompts, bs=self.embed_batch_size)
        np.savez_compressed(self.output_root / 'prompt_caption.npz', prompt_caption_ids=prompt_ids, prompt_caption=prompt_feats)

        # dataset_info.json
        info = {
            "dataset": "iu-xray",
            "num_samples": len(self.rows),
            "labels": LABELS_14,
            "image_dir": str(self.output_images),
        }
        with open(self.output_root / 'dataset_info.json', 'w', encoding='utf-8') as f:
            json.dump(info, f, ensure_ascii=False, indent=2)

        print(f"✓ Wrote: {cap_path}, {index_path}, {label_path}, prompt_caption.npz, dataset_info.json")


def _encode_prompts_ids_and_embeddings_with_progress(indications: List[str], bs: int = 256) -> Tuple[np.ndarray, np.ndarray]:
    """返回 (token_ids[B,77], prompt_embeddings[B,77,512])。
    - ids 用于模型内部编码的路径
    - embeddings 用于直接提供高质量向量，兼容旧流程
    """
    try:
        import torch
        from model import open_clip
        device = "cuda" if torch.cuda.is_available() else "cpu"
        clip_arch = 'ViT-B-16-quickgelu'
        clip_model, _, _ = open_clip.create_model_and_transforms(clip_arch, pretrained='metaclip_fullcc')
        clip_model = clip_model.to(device).eval()
        tokenizer = open_clip.get_tokenizer(clip_arch)

        ids_list = []
        feats = []
        with torch.no_grad():
            for i in tqdm(range(0, len(indications), bs), total=(len(indications)+bs-1)//bs, desc="Encoding prompts", unit="batch"):
                caps = [_clean_text(s or "", 512) for s in indications[i:i+bs]]
                tok = tokenizer(caps).to(device)  # [B, 77]
                # 保存 ids（cpu numpy int32）
                ids_list.append(tok.detach().cpu().numpy().astype('int32'))
                # 构造嵌入并通过 transformer
                x = clip_model.token_embedding(tok)
                x = x + clip_model.positional_embedding
                try:
                    x = clip_model.transformer(x, attn_mask=clip_model.attn_mask)
                except Exception:
                    x = clip_model.transformer(x)
                x_np = x.detach().cpu().numpy().astype('float32')
                # 截断/填充至 77
                if x_np.shape[1] > 77:
                    x_np = x_np[:, :77, :]
                elif x_np.shape[1] < 77:
                    pad = np.zeros((x_np.shape[0], 77 - x_np.shape[1], x_np.shape[2]), dtype='float32')
                    x_np = np.concatenate([x_np, pad], axis=1)
                feats.append(x_np)
        ids = np.concatenate(ids_list, axis=0)
        emb = np.concatenate(feats, axis=0)
        return ids, emb
    except Exception:
        # 仅生成 ids 的降级路径（不依赖模型权重下载）
        try:
            from model import open_clip
            clip_arch = 'ViT-B-16-quickgelu'
            tokenizer = open_clip.get_tokenizer(clip_arch)
            caps = [_clean_text(s or "", 512) for s in indications]
            tok = tokenizer(caps)  # [N,77]
            ids = tok.astype('int32')
            return ids, np.zeros((ids.shape[0], 77, 512), dtype=np.float32)
        except Exception:
            N = len(indications)
            return np.zeros((N, 77), dtype=np.int32), np.zeros((N, 77, 512), dtype=np.float32)


def main():
    import argparse
    from utils.image_path import iuxray_root
    parser = argparse.ArgumentParser(description="Convert IU-Xray for TriPAH.")
    parser.add_argument("--root", type=Path, default=iuxray_root)
    parser.add_argument("--output", type=Path, default=Path("dataset/iu-xray"))
    args = parser.parse_args()
    iu_root, out_root = args.root, args.output

    conv = IUXrayConverter(iu_root, out_root, image_resolution=384)
    conv.load_merge()
    conv.build_rows()
    conv.write_outputs()


if __name__ == '__main__':
    main()
