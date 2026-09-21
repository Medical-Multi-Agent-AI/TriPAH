import os
import sys
from pathlib import Path
import numpy as np
import scipy.io as sio


def validate(dataset_dir: Path) -> int:
    print(f"DATASET DIR: {dataset_dir}")
    if not dataset_dir.exists():
        print("ERROR: dataset dir missing")
        return 1

    # Files existence
    cap_path = dataset_dir / 'caption.txt'
    idx_path = dataset_dir / 'index.mat'
    lab_path = dataset_dir / 'label.mat'
    pc_path = dataset_dir / 'prompt_caption.npz'

    missing = [p.name for p in [cap_path, idx_path, lab_path, pc_path] if not p.exists()]
    if missing:
        print(f"ERROR: missing files: {missing}")
        return 1

    # caption.txt
    caps = [line.rstrip('\n') for line in open(cap_path, 'r', encoding='utf-8')]
    n_cap = len(caps)
    print("caption.txt lines:", n_cap)

    # index.mat
    idx = sio.loadmat(str(idx_path))
    print("index keys:", list(idx.keys()))
    index = idx.get('index')
    if index is None:
        print("ERROR: 'index' key missing in index.mat")
        return 1
    # Flatten and normalize: ensure paths are trimmed to avoid false missing
    index_flat = np.asarray([str(x).strip() for x in index.ravel()])
    N_index = len(index_flat)
    print("index count:", N_index)
    ok_prefix = all(str(x).startswith('images/') for x in index_flat)
    print("index prefix images/:", ok_prefix)
    # Check every referenced image. A partial check can miss an interrupted
    # Hub download near the end of the directory.
    exist_cnt = 0
    for x in index_flat:
        fp = dataset_dir / str(x).strip()
        exist_cnt += fp.exists()
    print("referenced images present:", f"{exist_cnt}/{N_index}")

    # label.mat
    lab = sio.loadmat(str(lab_path))
    print("label keys:", list(lab.keys()))
    labels = lab.get('labels')
    cats = lab.get('category')
    names = lab.get('label_names')
    print("labels shape:", None if labels is None else labels.shape)
    print("category shape:", None if cats is None else cats.shape)
    print("label_names len:", None if names is None else len(names.ravel()))

    # Label summary
    if labels is not None:
        N = labels.shape[0]
        label_names = [str(x) for x in names.ravel()] if names is not None else [f"L{i}" for i in range(labels.shape[1])]
        pos_counts = labels.sum(axis=0).astype(int)
        print("\nLabel distribution (count, percent):")
        for nm, cnt in sorted(zip(label_names, pos_counts.tolist()), key=lambda x: x[1], reverse=True):
            pct = (cnt / N) * 100.0
            print(f" - {nm:24s}: {cnt:6d} ({pct:5.1f}%)")
        # rows with all-zero labels
        all_zero = int((labels.sum(axis=1) == 0).sum())
        print(f"Rows with all-zero labels: {all_zero}/{N} ({(all_zero/N)*100.0:5.1f}%)")
        # No Finding count (if present)
        if names is not None:
            try:
                nf_idx = [i for i, nm in enumerate(label_names) if nm.lower() == 'no finding'][0]
                nf_cnt = int(labels[:, nf_idx].sum())
                print(f"No Finding positives: {nf_cnt}/{N} ({(nf_cnt/N)*100.0:5.1f}%)")
            except Exception:
                pass
        # sample a few rows
        print("\nSample rows (index, caption[:96], label vector sum, first 6 dims):")
        for i in range(min(5, N)):
            lv = labels[i]
            print(f" - {i:4d} | {caps[i][:96]!r} | sum={int(lv.sum())} | head={lv[:6].tolist()}")

    # prompt_caption.npz
    pc = np.load(str(pc_path), allow_pickle=True)
    print("prompt_caption.npz keys:", list(pc.files))
    arr = pc.get('prompt_caption')
    print("prompt_caption shape:", None if arr is None else arr.shape)

    # consistency
    N_pc = None if arr is None else arr.shape[0]
    N_labels = None if labels is None else labels.shape[0]
    print('N consistency:', {'caption': n_cap, 'index': N_index, 'labels': N_labels, 'prompt': N_pc})

    ok = (
        n_cap == N_index == N_labels == N_pc and
        ok_prefix and
        exist_cnt == N_index and
        arr is not None and arr.shape[1:] == (77, 512)
    )
    print('STATUS:', 'OK' if ok else 'ISSUES FOUND')
    return 0 if ok else 2


if __name__ == '__main__':
    ds = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('./dataset/iu-xray')
    code = validate(ds)
    sys.exit(code)
