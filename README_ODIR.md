# ODIR 数据准备与运行

TriPAH 将每位患者的左右眼图像裁去黑边后水平拼接，使用年龄、性别和双眼诊断关键词生成文本，并保留 ODIR 的八维多标签。转换器只读取有标签的训练图像；不会读取原始测试图像，也不会在转换阶段保留官方训练/测试划分。

## 数据路径与转换

先按主 [README](README.md) 安装依赖。原始数据目录需要包含 `data.xlsx` 和 `Training_Images/` 或 `Training Images/`。

```bash
export ODIR_ROOT=/path/to/ODIR-5K
python tools/validate_odir_setup.py --odir-root "$ODIR_ROOT"
python -m dataset.builders.convert_odir_dataset --root "$ODIR_ROOT" --output dataset/odir
```

自定义图像目录可分别传给两个命令的 `--train-dir /path/to/images`。也可设置 `TRIPAH_DATA_ROOT=/path/to/data`，此时默认 ODIR 路径为 `$TRIPAH_DATA_ROOT/ODIR/ODIR-5K`。转换器生成：

```text
dataset/odir/
├── images/
├── caption.txt
├── index.mat
├── label.mat
├── prompt_caption.npz
└── dataset_info.json
```

转换后检查样本数量、图像配对和提示特征。原始数据和生成的数据均不包含在代码仓库中。

## 训练与评估

下面的参数是运行示例，不代表已重新复现论文结果。专用脚本的默认值为 `seed=42`、`query-num=351`、`train-num=2275`，命令行参数可覆盖这些默认值。

```bash
python train_odir.py --k-bits-list 32,64,128 --epochs 100 --batch-size 32 \
  --query-num 351 --train-num 2275 --seed 42
```

训练日志和 checkpoint 保存在 `result/Result_TriPAH_ODIR/`。评估时保持与训练相同的哈希位数、样本划分和模型消融参数：

```bash
python main.py --dataset odir --caption-file caption.txt \
  --pretrained /path/to/model_best.pth --k-bits-list 32,64,128 \
  --query-num 351 --train-num 2275 --seed 42 --result-name results
```

数据加载器以指定 seed 对转换后的样本随机排列，先选 query，再从其余样本选 train；检索库包含所有非 query 样本，也包含训练样本。文本与提示包含诊断关键词，因此应在报告实验时明确这一输入条件。需要分析文本标签信息的影响时，可使用 `dataset/builders/create_odir_leakage_variants.py` 生成对照文本和提示，并通过 `--caption-file`、`--prompt-caption-file` 选择相应变体。

本地 MetaCLIP 权重可通过 `--clip-pretrained /path/to/open_clip_pytorch_model.bin` 指定。`--disable-fuse-mamba`、`--disable-prompt`、`--disable-recon` 控制相应消融。运行 `python main.py --help` 查看完整参数。
