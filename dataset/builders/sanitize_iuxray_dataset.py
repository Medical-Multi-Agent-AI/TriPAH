#!/usr/bin/env python3
"""
清理 IU-Xray 数据集：
- 移除 caption 为空/NaN/"nan" 的样本
- 同步过滤 index.mat、label.mat、prompt_caption.npz（含 ids/emb 及患者上下文）
- 保持四个文件的样本数一致，避免加载时出错
"""

import os
import sys
from pathlib import Path
import numpy as np
import scipy.io as sio


def _load_index_mat(idx_path: Path) -> np.ndarray:
    m = sio.loadmat(str(idx_path))
    arr = None
    for k in ("index", "imgs", "FAll"):
        if k in m:
            arr = m[k]
            break
    if arr is None:
        raise RuntimeError("index.mat 不包含 [index|imgs|FAll] 键")
    # 展平成 1D 字符串数组
    if isinstance(arr, np.ndarray):
        if arr.ndim == 2:
            if arr.shape[0] == 1:
                arr = arr[0]
            elif arr.shape[1] == 1:
                arr = arr[:, 0]
        arr = np.asarray([str(x).strip() for x in arr.ravel()])
    return arr


def _load_labels(lab_path: Path) -> tuple[np.ndarray, list[str]]:
    m = sio.loadmat(str(lab_path))
    arr = None
    for k in ("labels", "category", "label"):
        if k in m:
            arr = m[k]
            break
    if arr is None:
        raise RuntimeError("label.mat 不包含 [labels|category|label] 键")
    names = m.get("label_names")
    if names is not None:
        names = [str(x) for x in names.ravel()]
    else:
        names = [f"L{i}" for i in range(arr.shape[1])] if arr.ndim == 2 else []
    return arr.astype(np.float32), names


def _save_index(idx_path: Path, index: np.ndarray):
    max_len = max((len(p) for p in index), default=1)
    arr = np.array(index, dtype=f"<U{max_len}").reshape(-1, 1)
    sio.savemat(str(idx_path), {"index": arr}, do_compression=True)


def _save_labels(lab_path: Path, labels: np.ndarray, names: list[str]):
    sio.savemat(
        str(lab_path),
        {"labels": labels.astype(np.float32),
         "category": labels.astype(np.float32),
         "label_names": np.array(names, dtype=object)},
        do_compression=True,
    )


def sanitize(dataset_dir: Path) -> int:
    cap_path = dataset_dir / 'caption.txt'
    idx_path = dataset_dir / 'index.mat'
    lab_path = dataset_dir / 'label.mat'
    pc_path = dataset_dir / 'prompt_caption.npz'

    for p in (cap_path, idx_path, lab_path, pc_path):
        if not p.exists():
            print(f"ERROR: 缺少文件 {p}")
            return 1

    captions = [line.rstrip('\n') for line in open(cap_path, 'r', encoding='utf-8')]
    index = _load_index_mat(idx_path)
    labels, names = _load_labels(lab_path)
    pc = np.load(str(pc_path), allow_pickle=True)

    # 支持的 prompt 键
    prompt_ids = pc.get('prompt_caption_ids')
    prompt_emb = pc.get('prompt_caption')
    patient_ids = pc.get('patient_context_ids')
    patient_emb = pc.get('patient_context')

    N = len(captions)
    if not (len(index) == N and labels.shape[0] == N and ((prompt_ids is not None and prompt_ids.shape[0] == N) or (prompt_emb is not None and prompt_emb.shape[0] == N))):
        print("WARNING: 文件样本数不一致，仍将基于 caption 对齐裁剪")

    def _is_bad(s: str) -> bool:
        if s is None:
            return True
        t = str(s).strip().lower()
        return (t == '' or t in {'nan', 'none', 'null'})

    keep_mask = np.array([not _is_bad(c) for c in captions], dtype=bool)
    keep_idx = np.nonzero(keep_mask)[0]

    removed = int((~keep_mask).sum())
    print(f"将移除 {removed} 条无效 caption（空/NaN/\"nan\"）")
    if removed == 0:
        print("未发现需要移除的样本。")
        return 0

    # 过滤并回写
    captions_new = [captions[i] for i in keep_idx]
    with open(cap_path, 'w', encoding='utf-8') as f:
        for s in captions_new:
            f.write(s + '\n')

    index_new = index[keep_idx]
    _save_index(idx_path, index_new)

    labels_new = labels[keep_idx]
    _save_labels(lab_path, labels_new, names)

    save_kwargs = {}
    if prompt_ids is not None:
        save_kwargs['prompt_caption_ids'] = prompt_ids[keep_idx]
    if prompt_emb is not None:
        save_kwargs['prompt_caption'] = prompt_emb[keep_idx]
    if patient_ids is not None:
        save_kwargs['patient_context_ids'] = patient_ids[keep_idx]
    if patient_emb is not None:
        save_kwargs['patient_context'] = patient_emb[keep_idx]
    np.savez_compressed(str(pc_path), **save_kwargs)

    print("清理完成。样本数变更：")
    print(f"  caption: {len(captions)} -> {len(captions_new)}")
    print(f"  index:   {len(index)} -> {len(index_new)}")
    print(f"  labels:  {labels.shape[0]} -> {labels_new.shape[0]}")
    if prompt_ids is not None:
        print(f"  prompt_ids: {prompt_ids.shape[0]} -> {save_kwargs['prompt_caption_ids'].shape[0]}")
    if prompt_emb is not None:
        print(f"  prompt_emb: {prompt_emb.shape[0]} -> {save_kwargs['prompt_caption'].shape[0]}")
    return 0


if __name__ == '__main__':
    ds = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('./dataset/iu-xray')
    rc = sanitize(ds)
    sys.exit(rc)