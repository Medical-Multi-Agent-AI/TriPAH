#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."

CLIP_ARGS=()
if [[ -n "${CLIP_PRETRAINED:-}" ]]; then
  CLIP_ARGS=(--clip-pretrained "$CLIP_PRETRAINED")
fi

# # A: 基础架构（不包含任何创新点）
# python train_odir.py --k-bits-list 32,64,128 --epochs 100 --batch-size 128 --use-amp true --amp-dtype bf16 \
#   --disable-fuse-mamba --disable-prompt --disable-recon \
#   --hyper-global 0 --hyper-local 0 --hyper-text-prompt 0 --hyper-cbce 0 --hyper-cls-intra 0 --hyper-cls-inter 0 --hyper-quan 0 \
#   "${CLIP_ARGS[@]}" \
#   --exp-tag odir_A_basic

# A+B: 在 A 上增加三视图（开启提示与融合 + 图像/文本/提示对齐；仍不引入CBCE/贝叶斯/重建/量化）
# python train_odir.py --k-bits-list 32,64,128 --epochs 100 --batch-size 128 --use-amp true --amp-dtype bf16 \
#   --hyper-global 5 --hyper-local 5 --hyper-text-prompt 1.0 --tao-global 0.05 --tao-local 0.05 \
#   --disable-recon --hyper-cbce 0 --hyper-cls-intra 0 --hyper-cls-inter 0 --hyper-quan 0 \
#   "${CLIP_ARGS[@]}" \
#   --exp-tag odir_AB_triview

# A+C: 在 A 上增加不均衡监督（仅开启CBCE；保持提示/融合/对齐/贝叶斯/重建/量化为关闭）
python train_odir.py --k-bits-list 32,64,128 --epochs 100 --batch-size 128 --use-amp true --amp-dtype bf16 \
  --disable-fuse-mamba --disable-prompt --disable-recon \
  --hyper-global 0 --hyper-local 0 --hyper-text-prompt 0 --hyper-cbce 0.3 --hyper-cls-intra 0 --hyper-cls-inter 0 --hyper-quan 0 \
  "${CLIP_ARGS[@]}" \
  --exp-tag odir_AC_cbce

# A+B+C: 全部（Tri-view + CBCE；仍不引入贝叶斯/重建/量化）
python train_odir.py --k-bits-list 32,64,128 --epochs 100 --batch-size 128 --use-amp true --amp-dtype bf16 \
  --hyper-global 5 --hyper-local 5 --hyper-text-prompt 1.0 --tao-global 0.05 --tao-local 0.05 --hyper-cbce 0.3 \
  --disable-recon --hyper-cls-intra 0 --hyper-cls-inter 0 --hyper-quan 0 \
  "${CLIP_ARGS[@]}" \
  --exp-tag odir_ABC_all
