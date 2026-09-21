#!/usr/bin/env python3
"""
升级数据集中的 prompt_caption 嵌入：

- 读取 dataset/<name>/prompt_caption.npz
- 将其中的 prompt_caption [N,77,512] 通过 CLIP 文本 Transformer 前向编码，得到更高质量的 token 表示
- 保留已有的 prompt_caption_ids（如存在）以及患者上下文键
- 回写到原文件（np.savez_compressed）

注意：该脚本不需要原始指征文本，仅对已有的 token+pos 嵌入做深层编码升级。
"""

import os
import sys
from pathlib import Path
import numpy as np
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

def upgrade_prompt_embeddings(dataset_dir: Path, batch_size: int = 256) -> int:
    from model import open_clip
    import torch

    npz_path = dataset_dir / 'prompt_caption.npz'
    if not npz_path.exists():
        print(f"ERROR: 未找到 {npz_path}")
        return 1

    pc = np.load(str(npz_path), allow_pickle=False)
    emb = pc.get('prompt_caption')
    ids = pc.get('prompt_caption_ids')  # 可能不存在
    patient_ids = pc.get('patient_context_ids')
    patient_emb = pc.get('patient_context')
    if emb is None:
        print("ERROR: 文件中不包含 'prompt_caption'，无法升级")
        return 1

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    clip_arch = 'ViT-B-16-quickgelu'
    clip_model, _, _ = open_clip.create_model_and_transforms(clip_arch, pretrained='metaclip_fullcc')
    clip_model = clip_model.to(device).eval()

    N = emb.shape[0]
    width = emb.shape[-1]
    if emb.shape[1] != 77 or width != clip_model.ln_final.weight.shape[0]:
        print(f"WARNING: 发现非常规形状 {emb.shape}，将尝试截断/填充至 [N,77,{clip_model.ln_final.weight.shape[0]}]")
        # 调整到 77 长度
        if emb.shape[1] > 77:
            emb = emb[:, :77, :]
        elif emb.shape[1] < 77:
            pad = np.zeros((emb.shape[0], 77 - emb.shape[1], emb.shape[2]), dtype=emb.dtype)
            emb = np.concatenate([emb, pad], axis=1)

    upgraded = []
    with torch.no_grad():
        for i in tqdm(range(0, N, batch_size), total=(N + batch_size - 1)//batch_size, desc='Upgrading prompts', unit='batch'):
            x = torch.from_numpy(emb[i:i+batch_size]).to(device)
            # 已含 token+pos，因此直接经 transformer
            try:
                x = clip_model.transformer(x, attn_mask=clip_model.attn_mask)
            except Exception:
                x = clip_model.transformer(x)
            x_np = x.detach().cpu().numpy().astype('float32')
            upgraded.append(x_np)

    upgraded_emb = np.concatenate(upgraded, axis=0)
    save_kwargs = {'prompt_caption': upgraded_emb}
    if ids is not None:
        save_kwargs['prompt_caption_ids'] = ids
    if patient_ids is not None:
        save_kwargs['patient_context_ids'] = patient_ids
    if patient_emb is not None:
        save_kwargs['patient_context'] = patient_emb

    np.savez_compressed(str(npz_path), **save_kwargs)
    print(f"✓ 已升级 prompt_caption 嵌入并写回 {npz_path}")
    return 0


if __name__ == '__main__':
    ds = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('./dataset/iu-xray')
    rc = upgrade_prompt_embeddings(ds)
    sys.exit(rc)
