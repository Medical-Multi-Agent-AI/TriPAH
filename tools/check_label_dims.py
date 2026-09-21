#!/usr/bin/env python3
import os
from pathlib import Path
import scipy.io as scio
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
os.chdir(REPO_ROOT)

def check_label_dimensions():
    print("=== 检查iu-xray数据集标签维度 ===")

    # 检查label.mat文件
    try:
        label_data = scio.loadmat('dataset/iu-xray/label.mat')
        print(f"label.mat中的键: {list(label_data.keys())}")

        for key, value in label_data.items():
            if not key.startswith('__'):
                if hasattr(value, 'shape'):
                    print(f"{key}: shape={value.shape}, dtype={value.dtype}")
                    if len(value.shape) == 2 and value.shape[0] <= 10:
                        print(f"  前几行: {value[:min(5, value.shape[0])]}")
                else:
                    print(f"{key}: type={type(value)}")
    except Exception as e:
        print(f"读取label.mat失败: {e}")

    # 检查index.mat文件
    try:
        index_data = scio.loadmat('dataset/iu-xray/index.mat')
        print(f"\nindex.mat中的键: {list(index_data.keys())}")

        for key, value in index_data.items():
            if not key.startswith('__'):
                if hasattr(value, 'shape'):
                    print(f"{key}: shape={value.shape}, dtype={value.dtype}")
                else:
                    print(f"{key}: type={type(value)}")
    except Exception as e:
        print(f"读取index.mat失败: {e}")

    # 检查prompt_caption.npz文件
    try:
        prompt_data = np.load('dataset/iu-xray/prompt_caption.npz')
        print(f"\nprompt_caption.npz中的键: {list(prompt_data.keys())}")

        for key in prompt_data.keys():
            value = prompt_data[key]
            print(f"{key}: shape={value.shape}, dtype={value.dtype}")
            if key == 'prompt_caption' and len(value.shape) == 1:
                print(f"  前几个样本: {value[:3]}")
    except Exception as e:
        print(f"读取prompt_caption.npz失败: {e}")

if __name__ == "__main__":
    check_label_dimensions()
