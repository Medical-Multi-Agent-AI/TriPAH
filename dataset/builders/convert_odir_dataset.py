#!/usr/bin/env python3
"""
ODIR数据集转换脚本（只使用有标签的 Training Images）
将ODIR眼科疾病数据集转换为 TriPAH 兼容格式；数据集划分不在此脚本中进行，训练阶段由 load_data.py 自动完成。
"""

import os
import sys
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as scio
import re
from PIL import Image, ImageOps
import cv2

# Allow running this file directly from dataset/builders.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.image_path import odir_train_img_path, odir_test_img_path  # 测试集路径仅用于提示，不做读取


class ODIRConverter:
    def __init__(self, odir_root_path: str, output_path: str, use_wsl_paths: bool = True):
        self.odir_root = Path(odir_root_path)
        self.output_path = Path(output_path)
        self.excel_file = self.odir_root / "data.xlsx"

        if use_wsl_paths:
            self.training_images = Path(odir_train_img_path)
            self.testing_images = Path(odir_test_img_path)
            print("🔧 使用WSL路径配置:")
            print(f"   训练图像: {self.training_images}")
            print(f"   测试图像: {self.testing_images}（不使用）")
        else:
            self.training_images = self.odir_root / "Training Images"
            self.testing_images = self.odir_root / "Testing Images"

        self.disease_labels = {
            'N': 'Normal', 'D': 'Diabetes', 'G': 'Glaucoma', 'C': 'Cataract',
            'A': 'Age related Macular Degeneration', 'H': 'Hypertension',
            'M': 'Pathological Myopia', 'O': 'Other diseases/abnormalities'
        }

        self.output_path.mkdir(parents=True, exist_ok=True)
        (self.output_path / "images").mkdir(exist_ok=True)

    def load_excel_data(self) -> pd.DataFrame:
        print("📊 加载 ODIR 数据集 Excel 文件...")
        if not self.excel_file.exists():
            raise FileNotFoundError(f"Excel 文件不存在: {self.excel_file}")
        df = pd.read_excel(self.excel_file)
        print(f"✅ 成功加载 {len(df)} 条患者记录")
        return df

    # ===== 句式与语义解析辅助函数 =====
    def _normalize_sex(self, sex):
        s = str(sex).strip().lower() if pd.notna(sex) else ''
        if s in ['m', 'male']:
            return 'male'
        if s in ['f', 'female', 'woman']:
            return 'female'
        return 'patient'

    def _age_to_str(self, age):
        return str(int(age)) if pd.notna(age) else 'unknown'

    def _split_keywords(self, text):
        if not pd.notna(text):
            return []
        s = str(text).strip()
        if s == '':
            return []
        s = s.replace('|', ',')
        parts = re.split(r'[;,/]+|\band\b', s, flags=re.IGNORECASE)
        parts = [p.strip() for p in parts if p.strip()]
        return parts

    def _kw_is_normal(self, text):
        t = str(text).lower() if pd.notna(text) else ''
        return any(x in t for x in ['normal', 'normal fundus', 'no apparent abnormality', 'no obvious abnormality'])

    def _kw_has_laser(self, text):
        t = str(text).lower() if pd.notna(text) else ''
        return any(x in t for x in ['laser', 'photocoagulation', 'prp', 'panretinal'])

    def _region_hint(self, text):
        t = str(text).lower() if pd.notna(text) else ''
        if any(x in t for x in ['macula', 'fovea']):
            return 'macula'
        if any(x in t for x in ['disc', 'optic disc', 'cup', 'papilla']):
            return 'optic_disc'
        if any(x in t for x in ['vessel', 'artery', 'vein', 'neovascular']):
            return 'vessels'
        return None

    def _remove_black_borders(self, image_path: Path) -> np.ndarray:
        """
        去除眼底照片的黑边
        """
        try:
            # 使用PIL读取图像
            img = Image.open(image_path)
            img_array = np.array(img)

            # 转换为灰度图像用于边缘检测
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array

            # 使用阈值分割找到非黑色区域
            _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

            # 找到轮廓
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                # 找到最大的轮廓（通常是眼底图像的主体）
                largest_contour = max(contours, key=cv2.contourArea)

                # 获取边界框
                x, y, w, h = cv2.boundingRect(largest_contour)

                # 裁剪图像
                cropped = img_array[y:y+h, x:x+w]
                return cropped
            else:
                # 如果没有找到轮廓，返回原图像
                return img_array

        except Exception as e:
            print(f"⚠️ 图像黑边切除失败 {image_path}: {e}")
            # 如果处理失败，返回原图像
            img = Image.open(image_path)
            return np.array(img)

    def _concatenate_binocular_images(self, left_img_array: np.ndarray, right_img_array: np.ndarray) -> np.ndarray:
        """
        将左右眼图像水平拼接
        """
        try:
            # 确保两个图像的高度一致
            h1, w1 = left_img_array.shape[:2]
            h2, w2 = right_img_array.shape[:2]

            # 以较小的高度为准，调整图像大小
            target_height = min(h1, h2)

            # 调整左眼图像
            if h1 != target_height:
                aspect_ratio = w1 / h1
                target_width = int(target_height * aspect_ratio)
                left_resized = cv2.resize(left_img_array, (target_width, target_height))
            else:
                left_resized = left_img_array

            # 调整右眼图像
            if h2 != target_height:
                aspect_ratio = w2 / h2
                target_width = int(target_height * aspect_ratio)
                right_resized = cv2.resize(right_img_array, (target_width, target_height))
            else:
                right_resized = right_img_array

            # 水平拼接
            concatenated = np.concatenate([left_resized, right_resized], axis=1)
            return concatenated

        except Exception as e:
            print(f"⚠️ 图像拼接失败: {e}")
            # 如果拼接失败，返回左眼图像
            return left_img_array

    def _save_processed_image(self, img_array: np.ndarray, output_path: Path) -> bool:
        """
        保存处理后的图像
        """
        try:
            img = Image.fromarray(img_array)
            img.save(output_path, 'JPEG', quality=95)
            return True
        except Exception as e:
            print(f"⚠️ 图像保存失败 {output_path}: {e}")
            return False

    def _create_unified_description(self, age, sex, left_kw, right_kw) -> str:
        """
        创建统一格式的患者描述
        """
        age_str = self._age_to_str(age)
        sex_str = self._normalize_sex(sex).capitalize()

        # 处理左眼关键词
        left_desc = str(left_kw).strip() if pd.notna(left_kw) and str(left_kw).strip() else 'normal fundus'

        # 处理右眼关键词
        right_desc = str(right_kw).strip() if pd.notna(right_kw) and str(right_kw).strip() else 'normal fundus'

        # 生成统一格式的描述
        description = f"The Patient Age is {age_str}, The Patient Sex is {sex_str}, The Patient Left-Diagnostic Keywords is {left_desc}, The Patient Right-Diagnostic Keywords is {right_desc}."

        return description

    def _build_single_eye_caption(self, age, sex, eye, kw_text, kw_list, stage, condition, is_normal, is_laser, region_hint, asymmetry_hint=False):
        age_str = self._age_to_str(age)
        sex_str = self._normalize_sex(sex)
        eye_str = eye
        # 关键词短语（最多两项，使用 "with" 连接）
        if kw_list:
            if len(kw_list) == 1:
                kw_phrase = kw_list[0]
            else:
                kw_phrase = f"{kw_list[0]} with {kw_list[1]}"
        else:
            kw_phrase = str(kw_text).strip() if pd.notna(kw_text) and str(kw_text).strip() != '' else 'normal fundus'

        sentences = []
        if is_normal:
            sentences.append(f"The {eye_str} eye appears normal.")
        else:
            # 主句：含年龄性别与患侧
            sentences.append(f"A {age_str}-year-old {sex_str}; the {eye_str} eye fundus shows {kw_phrase}.")
            # 第二句优先使用 stage/condition 或区域提示/激光治疗/默认检查
            if (stage or condition) and (stage.strip() + condition.strip()).strip():
                sc = f"{stage}{condition}".strip()
                sentences.append(f"A {age_str}-year-old {sex_str} with {sc}; the {eye_str} eye shows {kw_phrase}.")
            elif is_laser:
                sentences.append(f"The {eye_str} eye shows {kw_phrase} after prior laser treatment.")
            elif region_hint == 'macula':
                sentences.append(f"Macular region of the {eye_str} eye shows {kw_phrase}.")
            elif region_hint in ['optic_disc', 'vessels']:
                sentences.append(f"Optic disc and vessels of the {eye_str} eye show {kw_phrase}.")
            else:
                sentences.append(f"The {eye_str} eye reveals {kw_phrase} on retinal examination.")
            if asymmetry_hint:
                sentences = [sentences[0], f"The {eye_str} eye has {kw_phrase}; the fellow eye requires evaluation."]
        # 限制为 1–2 句
        final = sentences[:2]
        return " ".join(s.strip() for s in final)

    def _copy_image_optimized(self, src_name: str, dst_name: str, train_files: set) -> dict:
        if not isinstance(src_name, str) or len(src_name) == 0:
            return {"success": False, "reason": "empty filename"}
        if src_name not in train_files:
            return {"success": False, "reason": "not in training set"}
        src = self.training_images / src_name
        if not src.exists():
            return {"success": False, "reason": "file missing"}
        dst = self.output_path / "images" / dst_name
        shutil.copyfile(src, dst)
        return {"success": True, "split": "train"}

    def _create_caption_file(self, captions: list):
        cap_file = self.output_path / "caption.txt"
        with open(cap_file, "w", encoding="utf-8") as f:
            for c in captions:
                f.write(str(c).strip() + "\n")
        print(f"✅ 创建文本文件: {cap_file}")

    def _create_index_file(self, image_paths: list):
        index_file = self.output_path / "index.mat"
        max_len = max(len(p) for p in image_paths) if image_paths else 1
        index_array = np.array(image_paths, dtype=f"<U{max_len}")
        scio.savemat(index_file, {"index": index_array})
        print(f"✅ 创建索引文件: {index_file} | 共 {len(image_paths)} 条")
        print("   注意: 数据划分将由 load_data.py 在训练阶段根据 query_num/train_num 自动完成")

    def _create_label_file(self, labels: list):
        label_file = self.output_path / "label.mat"
        labels_array = np.array(labels, dtype=np.float32)
        scio.savemat(label_file, {"category": labels_array, "labels": labels_array, "disease_names": list(self.disease_labels.values())})
        print(f"✅ 创建标签文件: {label_file}")

    def _create_prompt_caption_file(self, captions: list):
        prompt_file = self.output_path / "prompt_caption.npz"
        num = len(captions)
        max_words = 77
        dim = 512
        try:
            import torch
            from model import open_clip
            print("🔄 加载 CLIP 模型以生成文本特征...")
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            clip_model, _, _ = open_clip.create_model_and_transforms('ViT-B-16-quickgelu', pretrained='metaclip_fullcc')
            clip_model = clip_model.to(device)
            tokenizer = open_clip.get_tokenizer('ViT-B-16-quickgelu')
            clip_model.eval()
            feats = []
            bs = 64
            with torch.no_grad():
                for i in range(0, num, bs):
                    texts = [str(t) for t in captions[i:i+bs]]
                    tok = tokenizer(texts).to(device)
                    if hasattr(clip_model, 'text'):
                        x = clip_model.text.token_embedding(tok)
                        x = x + clip_model.text.positional_embedding
                        x = clip_model.text.transformer(x, attn_mask=clip_model.text.attn_mask)
                        x = clip_model.text.ln_final(x)
                    else:
                        x = clip_model.token_embedding(tok)
                        x = x + clip_model.positional_embedding
                        x = clip_model.transformer(x, attn_mask=clip_model.attn_mask)
                        x = clip_model.ln_final(x)
                    if x.size(1) < max_words:
                        pad = torch.zeros(x.size(0), max_words - x.size(1), dim, device=x.device, dtype=x.dtype)
                        x = torch.cat([x, pad], dim=1)
                    elif x.size(1) > max_words:
                        x = x[:, :max_words, :]
                    feats.append(x.cpu().numpy().astype(np.float32))
            arr = np.concatenate(feats, axis=0)
            np.savez(prompt_file, prompt_caption=arr)
            print(f"✅ 创建 prompt_caption.npz（CLIP特征）: {arr.shape}")
        except Exception as e:
            print(f"⚠️ 文本特征生成失败，使用零占位: {e}")
            arr = np.zeros((num, max_words, dim), dtype=np.float32)
            np.savez(prompt_file, prompt_caption=arr)
            print(f"✅ 创建 prompt_caption.npz（占位零特征）: {arr.shape}")

    def process_images_and_create_dataset(self, df: pd.DataFrame, preserve_original_split: bool = True) -> dict:
        print("🖼️  开始处理图像并创建数据文件（双眼拼接模式）...")
        all_captions, all_labels, image_paths = [], [], []

        print("🔍 预扫描训练集图像文件...")
        train_files = set(f.name for f in self.training_images.glob('*'))
        print(f"   训练集可用图像: {len(train_files)} 个")
        print(f"   测试集图像: 已忽略，不参与处理")

        total_rows = len(df)
        processed = 0
        successful_pairs = 0

        for idx, row in df.iterrows():
            processed += 1
            if processed % 200 == 0 or processed == total_rows:
                print(f"   进度: {processed}/{total_rows} ({processed/total_rows*100:.1f}%)")

            left_img = row.get('Left-Fundus')
            left_kw = row.get('Left-Diagnostic Keywords')
            right_img = row.get('Right-Fundus')
            right_kw = row.get('Right-Diagnostic Keywords')
            disease_vec = [row.get(k, 0) for k in ['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']]

            # 检查左右眼图像是否都存在
            if not (isinstance(left_img, str) and isinstance(right_img, str)):
                continue
            if left_img not in train_files or right_img not in train_files:
                continue

            try:
                # 处理左眼图像：去除黑边
                left_img_path = self.training_images / left_img
                left_processed = self._remove_black_borders(left_img_path)

                # 处理右眼图像：去除黑边
                right_img_path = self.training_images / right_img
                right_processed = self._remove_black_borders(right_img_path)

                # 拼接左右眼图像
                binocular_img = self._concatenate_binocular_images(left_processed, right_processed)

                # 保存拼接后的图像
                output_filename = f"patient_{idx}_binocular.jpg"
                output_path = self.output_path / "images" / output_filename

                if self._save_processed_image(binocular_img, output_path):
                    # 生成统一格式的描述
                    unified_description = self._create_unified_description(
                        row.get('Patient Age'),
                        row.get('Patient Sex'),
                        left_kw,
                        right_kw
                    )

                    # 添加到数据集
                    image_paths.append(f"images/{output_filename}")
                    all_captions.append(unified_description)
                    all_labels.append(disease_vec)
                    successful_pairs += 1

            except Exception as e:
                print(f"⚠️ 处理患者 {idx} 的图像时出错: {e}")
                continue

        print(f"✅ 成功处理 {successful_pairs} 对双眼图像")
        print(f"   总图像数: {len(image_paths)}")

        # 生成所需文件
        self._create_caption_file(all_captions)
        self._create_index_file(image_paths)
        self._create_label_file(all_labels)
        self._create_prompt_caption_file(all_captions)

        return {"total_samples": len(image_paths), "preserve_split": preserve_original_split, "binocular_pairs": successful_pairs}

    def create_dataset_info(self, stats: dict):
        info = {
            "dataset_name": "ODIR",
            "description": "Ocular Disease Intelligent Recognition Dataset (Binocular Mode)",
            "total_samples": stats.get('total_samples', 0),
            "binocular_pairs": stats.get('binocular_pairs', 0),
            "split_strategy": "No split here. load_data.py will randomly split into query/train/retrieval at training time.",
            "disease_labels": self.disease_labels,
            "image_format": "JPEG (Binocular concatenated)",
            "image_size": "Variable (Left-Right concatenated)",
            "processing": "Black border removal + Binocular concatenation",
            "description_format": "Unified patient description with age, sex, and bilateral diagnostic keywords",
            "created_by": "ODIR Converter Script (Binocular Mode)"
        }
        info_file = self.output_path / "dataset_info.json"
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(info, f, indent=2, ensure_ascii=False)
        print(f"✅ 创建数据集信息文件: {info_file}")


def main():
    print("=" * 60)
    print("🏥 ODIR 数据集转换器（只使用 Training Images）")
    print("=" * 60)

    import argparse
    from utils.image_path import odir_root as default_root
    parser = argparse.ArgumentParser(description="Convert ODIR binocular images for TriPAH.")
    parser.add_argument("--root", type=Path, default=default_root)
    parser.add_argument("--output", default="dataset/odir")
    parser.add_argument("--train-dir", type=Path, default=None)
    args = parser.parse_args()
    odir_root, output_path = args.root, args.output

    try:
        converter = ODIRConverter(odir_root, output_path, use_wsl_paths=False)
        if args.train_dir:
            converter.training_images = args.train_dir
        elif os.environ.get("ODIR_TRAIN_IMAGES"):
            converter.training_images = Path(odir_train_img_path)
        elif (odir_root / "Training_Images").exists():
            converter.training_images = odir_root / "Training_Images"
        df = converter.load_excel_data()
        stats = converter.process_images_and_create_dataset(df, preserve_original_split=True)
        converter.create_dataset_info(stats)

        print("\n🎉 ODIR 数据集转换完成（双眼拼接模式）！")
        print(f"📁 输出目录: {output_path}")
        print("\n📋 生成的文件:")
        print("   - images/                # 双眼拼接图像文件夹")
        print("   - caption.txt            # 统一格式患者描述")
        print("   - index.mat              # 图像索引（仅路径，划分由训练阶段处理）")
        print("   - label.mat              # 疾病标签")
        print("   - prompt_caption.npz     # 提示文本特征")
        print("   - dataset_info.json      # 数据集信息")

        print("\n📊 数据集统计:")
        print(f"   双眼拼接对数: {stats['binocular_pairs']}")
        print(f"   总样本数: {stats['total_samples']}")
        print("   训练/查询/检索划分: 由 load_data.py 在训练阶段根据 query_num/train_num 自动完成")
        print(f"   保持原始分割标志: {'是' if stats['preserve_split'] else '否'}（此脚本未做划分）")

        print("\n🔧 处理特性:")
        print("   - 眼底照片黑边自动切除")
        print("   - 左右眼图像水平拼接")
        print("   - 统一格式患者描述生成")
        print("   - 描述格式: 'The Patient Age is X, The Patient Sex is Y, The Patient Left-Diagnostic Keywords is Z, The Patient Right-Diagnostic Keywords is W.'")

        print("\n🚀 接下来的步骤:")
        print("   1. 训练模型: python train_odir.py --k-bits-list 64")
        print("   2. 测试模型: 参见 README_ODIR.md 中的评估命令")
    except Exception as e:
        print(f"❌ 转换失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
