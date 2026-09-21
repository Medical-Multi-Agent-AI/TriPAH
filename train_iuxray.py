#!/usr/bin/env python3
"""
IU-Xray 数据集训练脚本
使用 TriPAH 模型训练 IU-Xray 胸部X光数据集（14 类，保留 0.5 标签）
"""

import os
import sys
import argparse
import torch
import numpy as np
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent))

from load_data import generate_dataset
import utils.calc_utils as calc_utils
from utils.get_args import get_args


def setup_iuxray_training():
    """Apply dataset defaults while honoring explicit command-line options."""
    return get_args(defaults={
        'dataset': 'iu-xray',
        'k_bits_list': '32,64,128',
        'query_num': 380,
        'train_num': 2500,
        'caption_file': 'caption.txt',
        'seed': 42,
        'num_workers': 24,
        'lr': 0.001,
        'weight_decay': 0.0001,
        'epochs': 100,
        'valid_freq': 2,
        'gpu_rank': 0,
        'recon': 0.001,
        'hyper_cls_inter': 5.0,
        'hyper_quan': 0.8,
        'accumulation_steps': 2,
        'data_path': 'dataset/iu-xray',
        'result_name': 'result/Result_TriPAH_IUXray',
        'device': 'cuda',
        'is_train': True,
    })


def check_iuxray_dataset(data_path):
    """检查 IU-Xray 数据集是否存在"""
    data_path = Path(data_path)

    required_files = [
        'caption.txt',
        'index.mat',
        'label.mat',
        'prompt_caption.npz',
        'images'
    ]

    missing_files = []
    for file_name in required_files:
        file_path = data_path / file_name
        if not file_path.exists():
            missing_files.append(file_name)

    if missing_files:
        print("❌ IU-Xray 数据集文件缺失:")
        for file_name in missing_files:
            print(f"   - {file_name}")
        print("\n💡 请先运行转换脚本:")
        print("   python -m dataset.builders.convert_iuxray_dataset")
        return False

    print("✅ IU-Xray 数据集文件检查通过")
    return True


def load_iuxray_dataset(args):
    """加载 IU-Xray 数据集"""
    print("📊 加载 IU-Xray 数据集...")

    try:
        # 解析数据文件路径：若为绝对路径或已存在则直接使用；否则拼接到 dataset/iu-xray 下
        def _resolve(p: str) -> str:
            try:
                if os.path.isabs(p) or os.path.exists(p):
                    return p
            except Exception:
                pass
            return os.path.join("dataset", "iu-xray", p)

        caption_path = _resolve(args.caption_file)
        index_path = _resolve(getattr(args, "index_file", "index.mat"))
        label_path = _resolve(getattr(args, "label_file", "label.mat"))

        train_data, query_data, retrieval_data = generate_dataset(
            captionFile=caption_path,
            indexFile=index_path,
            labelFile=label_path,
            dataset_name=args.dataset,
            maxWords=args.max_words,
            imageResolution=args.resolution,
            query_num=args.query_num,
            train_num=args.train_num,
            seed=args.seed
        )

        print(f"✅ 数据集加载成功:")
        print(f"   训练集: {len(train_data)} 样本")
        print(f"   查询集: {len(query_data)} 样本")
        print(f"   检索集: {len(retrieval_data)} 样本")

        return train_data, query_data, retrieval_data

    except Exception as e:
        print(f"❌ 数据集加载失败: {e}")
        return None, None, None


def train_iuxray_model(args, train_data, query_data, retrieval_data):
    """训练 IU-Xray 模型"""
    print("🚀 开始训练 IU-Xray 模型...")

    # 创建 log_dir 基于 save_dir
    args.log_dir = os.path.join(args.save_dir, 'logs')

    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"🔧 使用设备: {device}")

    try:
        # 使用 TrainerAsym 进行训练
        print("⏳ 初始化 TrainerAsym 训练器...")
        from train_asym import TrainerAsym
        args.is_train = True
        trainer = TrainerAsym(args)

        print("✅ 训练器初始化成功")
        print(f"📊 训练参数:")
        print(f"   批次大小: {args.batch_size}")
        print(f"   学习率: {args.lr}")
        print(f"   训练轮数: {args.epochs}")
        print(f"   哈希位数列表: {args.k_bits_list}")

        print("🎯 模型训练已完成")
        return True

    except Exception as e:
        print(f"❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def evaluate_iuxray_model(args, query_data, retrieval_data):
    """评估 IU-Xray 模型"""
    print("📈 评估模型性能...")

    # 检查是否有训练好的模型
    model_files = []
    if os.path.exists(args.save_dir):
        for file in os.listdir(args.save_dir):
            if file.endswith('.pth'):
                model_files.append(os.path.join(args.save_dir, file))

    if not model_files:
        print("⚠️  未找到训练好的模型文件，跳过评估")
        return True

    # 使用最新的模型文件
    latest_model = max(model_files, key=os.path.getctime)
    print(f"📂 使用模型文件: {latest_model}")

    try:
        from train_asym import TrainerAsym
        import copy
        test_args = copy.deepcopy(args)
        test_args.is_train = False
        test_args.pretrained = latest_model
        trainer = TrainerAsym(test_args)
        print("✅ 模型评估完成")
        return True

    except Exception as e:
        print(f"❌ 评估失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("=" * 60)
    print("🩺 IU-Xray 数据集训练 - TriPAH")
    print("=" * 60)

    args = setup_iuxray_training()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if not check_iuxray_dataset(args.data_path):
        return

    train_data, query_data, retrieval_data = load_iuxray_dataset(args)
    if train_data is None:
        return

    training_success = train_iuxray_model(args, train_data, query_data, retrieval_data)

    if training_success:
        print("\n" + "=" * 60)
        print("🔍 开始模型评估")
        print("=" * 60)
        evaluate_iuxray_model(args, query_data, retrieval_data)
    else:
        print("\n⚠️  由于训练失败，跳过模型评估")

    print("\n🎉 IU-Xray 训练流程完成！")


if __name__ == "__main__":
    main()
