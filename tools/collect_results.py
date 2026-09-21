#!/usr/bin/env python3
import os, re, sys, json
import glob
import scipy.io as scio
import numpy as np
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
os.chdir(REPO_ROOT)

def parse_tag_from_dir(p: str):
    m = re.search(r"_(odir|iux|mimic)_[A-Z]+", p)
    return None

def collect(dataset: str, results_root: str = "results"):
    rows = []
    search_roots = [
        Path(results_root),
        Path("result/Result_TriPAH_ODIR"),
        Path("result/Result_TriPAH_IUXray"),
        Path("result/Result_TriPAH_MIMIC"),
    ]
    mats = []
    for root in search_roots:
        if root.exists():
            mats += glob.glob(str(root / "**" / f"TriPAH-{dataset}-*bit-*-test.mat"), recursive=True)
    for mat in mats:
        fn = Path(mat).name
        m = re.match(rf"TriPAH-{dataset}-(\d+)bit-(.+)-test\.mat", fn)
        if not m:
            continue
        bits = int(m.group(1))
        tag = m.group(2)
        # 推断 test.log: 位于 mat 文件的父目录的上一级
        # path like: <save_dir>/mat_results/<fn> -> test log: <save_dir>/test.log
        test_log = Path(mat).parent.parent / "test.log"
        if not test_log.exists():
            test_log = None

        map_i2t = map_t2i = avg_map = ""
        if test_log:
            try:
                with open(test_log, 'r', encoding='utf-8') as f:
                    txt = f.read()
                # 简单正则：MAP\(i->t\): x.xx, MAP\(t->i\): y.yy, avg mAP: z.zz
                mi = re.search(rf"{bits} bits: MAP\(i->t\)\s*[:=]\s*([0-9.]+)", txt)
                mt = re.search(rf"{bits} bits: MAP\(i->t\)[^\n]*MAP\(t->i\)\s*[:=]\s*([0-9.]+)", txt)
                ma = re.search(r"avg mAP\s*[:=]\s*([0-9.]+)", txt)
                if mi: map_i2t = mi.group(1)
                if mt: map_t2i = mt.group(1)
                if mi and mt: avg_map = (float(map_i2t) + float(map_t2i)) / 2
            except Exception:
                pass

        rows.append({
            "dataset": dataset,
            "tag": tag,
            "bits": bits,
            "map_i2t": map_i2t,
            "map_t2i": map_t2i,
            "avg_map": avg_map,
            "file": fn
        })
    return rows

def write_csv(rows, out_path):
    import csv
    keys = ["dataset","tag","bits","map_i2t","map_t2i","avg_map","best_epoch","epochs","batch_size","file"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})

def augment_from_logs(rows, result_dirs):
    # 读取各 save_dir/train.log，提取最优 epoch 与训练超参；匹配 tag
    for rd in result_dirs:
        for log in glob.glob(os.path.join(rd, "**", "train.log"), recursive=True):
            # 提取目录中的 tag（最后一段包含 _<tag>_timestamp）
            d = Path(log).parent.parent.name
            m = re.search(r"_(odir|iux|mimic)_[A-Z_]+", d)
            # 简化处理：跳过复杂匹配，保留空，由手工填入或后续完善
            pass
    return rows

def to_float(x):
    try:
        return float(x)
    except Exception:
        return None

def summarize(rows, out_path):
    # group by (dataset, tag), average over bits (32/64/128) when available
    from collections import defaultdict
    groups = defaultdict(list)
    for r in rows:
        groups[(r.get("dataset"), r.get("tag"))].append(r)

    import csv
    keys = ["dataset","tag","bits_set","map_i2t_mean","map_t2i_mean","avg_map_mean","count_bits"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for (ds, tag), items in groups.items():
            mi = [to_float(it.get("map_i2t")) for it in items]
            mt = [to_float(it.get("map_t2i")) for it in items]
            ma = [to_float(it.get("avg_map")) for it in items]
            mi = [v for v in mi if v is not None]
            mt = [v for v in mt if v is not None]
            ma = [v for v in ma if v is not None]
            mi_mean = sum(mi)/len(mi) if mi else ""
            mt_mean = sum(mt)/len(mt) if mt else ""
            ma_mean = sum(ma)/len(ma) if ma else ""
            bits_set = ",".join(sorted({str(it.get("bits")) for it in items}))
            w.writerow({
                "dataset": ds,
                "tag": tag,
                "bits_set": bits_set,
                "map_i2t_mean": mi_mean,
                "map_t2i_mean": mt_mean,
                "avg_map_mean": ma_mean,
                "count_bits": len(items)
            })

def main():
    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    detail_csv = out_dir / "ablation_detail.csv"
    summary_csv = out_dir / "ablation_summary.csv"

    all_rows = []
    all_rows += collect("odir")
    all_rows += collect("iu-xray")
    all_rows += collect("mimic-cxr")
    write_csv(all_rows, detail_csv)
    summarize(all_rows, summary_csv)
    print(f"[OK] Saved detail to {detail_csv}")
    print(f"[OK] Saved summary to {summary_csv}")

if __name__ == "__main__":
    main()
