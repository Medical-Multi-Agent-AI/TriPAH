from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

from model.open_clip.tokenizer import SimpleTokenizer
import os
import numpy as np
import scipy.io as scio

from torch.utils.data import Dataset
import torch
import random
from PIL import Image
from torchvision.transforms import Compose, RandomCrop, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode, RandomResizedCrop, RandomHorizontalFlip, RandomRotation, ColorJitter, RandomAutocontrast, RandomApply
from utils.image_path import *


class BaseDataset(Dataset):
    def __init__(self,
                 captions: dict,
                 indexs: dict,
                 labels: dict,
                 prompt_captions: dict,
                 prompt_embed: dict = None,
                 patient_context_ids: dict = None,
                 patient_context_emb: dict = None,
                 img_path: str = None,
                 dataset_name: str = None,
                 is_train=True,
                 tokenizer=SimpleTokenizer(),
                 maxWords=77,
                 imageResolution=224,
                 ):

        self.captions = captions
        self.prompt_captions = prompt_captions
        self.prompt_embed = prompt_embed
        self.patient_context_ids = patient_context_ids
        self.patient_context_emb = patient_context_emb
        self.indexs = indexs
        self.labels = labels
        self.maxWords = maxWords
        self.tokenizer = tokenizer
        self.img_path = img_path
        self.dataset_name = dataset_name
        self.is_train = is_train

        # ==== 眼科友好增广 ====
        if is_train:
            self.transform = Compose([
                RandomResizedCrop(
                    size=imageResolution,
                    scale=(0.80, 1.00),
                    interpolation=InterpolationMode.BICUBIC
                ),
                # 小角度旋转，避免左右语义翻转；保持病灶形态
                RandomApply([
                    RandomRotation(degrees=10, fill=(0, 0, 0))
                ], p=0.4),
                # 轻量光照/对比度扰动
                RandomApply([
                    ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05)
                ], p=0.5),
                # 自适应对比度（类似 CLAHE 的轻量替代）
                RandomAutocontrast(p=0.3),
                ToTensor(),
                Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
            ])
        else:
            self.transform = Compose([
                Resize(imageResolution, interpolation=InterpolationMode.BICUBIC),
                CenterCrop(imageResolution),
                ToTensor(),
                Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
            ])

        self.SPECIAL_TOKEN = {"CLS_TOKEN": "ffffffff", "SEP_TOKEN": "[SEP]", "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}

        self.__length = len(self.indexs)

    def __len__(self):
        return self.__length

    def _load_image(self, index: int) -> torch.Tensor:
        # 获取原始文件名
        original_filename = self.indexs[index].split("/")[-1].strip()

        # 根据数据集类型处理文件名
        if hasattr(self, 'dataset_name'):
            dataset_name = self.dataset_name
        else:
            # 通过img_path推断数据集类型
            if 'coco' in self.img_path.lower():
                dataset_name = 'coco'
            elif 'flickr' in self.img_path.lower():
                dataset_name = 'flickr25k'
            elif 'nuswide' in self.img_path.lower():
                dataset_name = 'nuswide'
            elif 'mimic-cxr' in self.img_path.lower():
                dataset_name = 'mimic-cxr'
            else:
                dataset_name = 'unknown'

        # 对于COCO数据集，需要添加前缀
        if dataset_name == 'coco':
            if not original_filename.startswith('COCO_'):
                # 提取数字部分
                base_name = os.path.splitext(original_filename)[0]
                ext = os.path.splitext(original_filename)[1]

                # 尝试不同的COCO前缀（训练集和验证集）
                possible_prefixes = ['COCO_train2014_', 'COCO_val2014_']
                filename = None

                for prefix in possible_prefixes:
                    test_filename = f"{prefix}{base_name.zfill(12)}{ext}"
                    test_path = os.path.join(self.img_path, test_filename)
                    if os.path.exists(test_path):
                        filename = test_filename
                        break

                # 如果都找不到，默认使用训练集前缀
                if filename is None:
                    filename = f"COCO_train2014_{base_name.zfill(12)}{ext}"
            else:
                filename = original_filename
            image_path = os.path.join(self.img_path, filename)
        elif dataset_name == 'odir':
            # ODIR数据集的图像文件存储在images子文件夹中
            filename = original_filename
            image_path = os.path.join(self.img_path, 'images', filename)
        elif dataset_name == 'mimic-cxr':
            # MIMIC-CXR数据集的图像文件存储在images子文件夹中，保持完整的层级结构
            # index.mat中的路径格式: p10/p10000032/s50414267/02aa804e-bde0afdd-112c0b34-7bc16630-4e384014.jpg
            # 需要拼接为: dataset/mimic-cxr/images/p10/p10000032/s50414267/02aa804e-bde0afdd-112c0b34-7bc16630-4e384014.jpg
            full_relative_path = self.indexs[index].strip()
            image_path = os.path.join(self.img_path, 'images', full_relative_path)
        elif dataset_name == 'iu-xray':
            # IU-Xray 的图像位于 images 子目录
            filename = original_filename
            image_path = os.path.join(self.img_path, 'images', filename)
        else:
            filename = original_filename
            image_path = os.path.join(self.img_path, filename)
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        return image

    def _load_text(self, index: int):
        captions = self.captions[index]
        use_cap = captions[random.randint(0, len(captions) - 1)]
        # 训练前：字符级截断至 512，防止极端长文本；随后交给 CLIP BPE 处理
        if isinstance(use_cap, bytes):
            use_cap = use_cap.decode('utf-8', errors='ignore')
        use_cap = use_cap[:512]
        # 直接使用 tokenizer 调用以自动添加 tresult, 0
        caption = self.tokenizer([use_cap], context_length=self.maxWords)[0]
        key_padding_mask = (caption == 0)
        return caption, key_padding_mask

    def _load_label(self, index: int) -> torch.Tensor:
        label = self.labels[index]
        label = torch.from_numpy(label)
        return label

    def get_all_label(self):
        # 保留浮点标签以支持 0/0.5/1，不进行整型截断
        labels = torch.zeros([self.__length, len(self.labels[0])], dtype=torch.float32)
        for i, item in enumerate(self.labels):
            labels[i] = torch.from_numpy(item).float()
        return labels

    def __getitem__(self, index):
        image = self._load_image(index)
        caption, key_padding_mask = self._load_text(index)
        label = self._load_label(index)

        # 文本提示：以 p=0.5 随机选择 主提示 或 患者级上下文（若存在）
        use_ctx = False
        if (self.patient_context_ids is not None or self.patient_context_emb is not None):
            use_ctx = (random.random() < 0.5)

        if self.prompt_captions is not None:  # 主提示为 ids
            if use_ctx and self.patient_context_ids is not None:
                prompt_caption = torch.from_numpy(self.patient_context_ids[index]).long()
            else:
                prompt_caption = torch.from_numpy(self.prompt_captions[index]).long()
        elif self.prompt_embed is not None:  # 主提示为嵌入向量
            if use_ctx and self.patient_context_emb is not None:
                prompt_caption = torch.from_numpy(self.patient_context_emb[index]).float()
            else:
                prompt_caption = torch.from_numpy(self.prompt_embed[index]).float()
        else:
            prompt_caption = None

        if prompt_caption is not None:
            return image, caption, key_padding_mask, label, prompt_caption, index
        else:
            return image, caption, key_padding_mask, label, index


def split_data(captions, indexs, labels, prompt_captions, query_num, train_num, seed=None,
               return_split_indices=False):
    np.random.seed(seed=seed)

    random_index = np.random.permutation(range(len(indexs)))
    query_index = random_index[: query_num]
    train_index = random_index[query_num: query_num + train_num]
    retrieval_index = random_index[query_num:]

    query_indexs = indexs[query_index]
    query_captions = captions[query_index]
    query_labels = labels[query_index]

    train_indexs = indexs[train_index]
    train_captions = captions[train_index]
    train_labels = labels[train_index]

    retrieval_indexs = indexs[retrieval_index]
    retrieval_captions = captions[retrieval_index]
    retrieval_labels = labels[retrieval_index]

    # 处理 prompt_captions，如果为 None 则返回 None 元组
    if prompt_captions is not None:
        query_prompt_captions = prompt_captions[query_index]
        train_prompt_captions = prompt_captions[train_index]
        retrieval_prompt_captions = prompt_captions[retrieval_index]
        split_prompt_captions = (query_prompt_captions, train_prompt_captions, retrieval_prompt_captions)
    else:
        split_prompt_captions = (None, None, None)

    split_indexs = (query_indexs, train_indexs, retrieval_indexs)
    split_captions = (query_captions, train_captions, retrieval_captions)
    split_labels = (query_labels, train_labels, retrieval_labels)

    result = (split_indexs, split_captions, split_labels, split_prompt_captions)
    if return_split_indices:
        return result + ((query_index, train_index, retrieval_index),)
    return result


def _split_optional_array(values, split_indices):
    """Apply the dataset split to an optional row-aligned NumPy array."""
    if values is None:
        return (None, None, None)
    return tuple(values[index] for index in split_indices)


def generate_dataset(captionFile: str,
                     indexFile: str,
                     labelFile: str,
                     dataset_name: str,
                     maxWords=77,
                     imageResolution=224,
                     query_num=2000,
                     train_num=10000,
                     promptCaptionFile: str = None,
                     seed=None,
                     ):
    npy = False
    if captionFile.endswith("mat"):

        captions = scio.loadmat(captionFile)
        if "caption" in captions:
            captions = captions["caption"]
        elif "tags" in captions:
            captions = captions["tags"]
        elif "YAll" in captions:
            captions = captions["YAll"]
        else:
            raise RuntimeError("text file is not support, we only read the keys of [caption, tags, YAll].")
        captions = captions[0] if captions.shape[0] == 1 else captions
    elif captionFile.endswith("txt"):
        with open(captionFile, 'r', encoding="utf-8") as f:
            captions = f.readlines()
        captions = np.asarray([[item.strip()] for item in captions])
    else:
        raise ValueError("the format of 'captionFile' doesn't support, only support [txt, mat] format.")

    if indexFile.endswith("mat"):
        npy = False
        _mat = scio.loadmat(indexFile)
        if "index" in _mat:
            indexs = _mat["index"]
        elif "imgs" in _mat:
            indexs = _mat["imgs"]
        elif "FAll" in _mat:
            indexs = _mat["FAll"]
        else:
            raise RuntimeError("image file is not support, we only read the keys of [caption, tags, YAll].")
        # Flatten 1xN/Nx1 matrices to 1D strings
        if isinstance(indexs, np.ndarray):
            if indexs.ndim == 2:
                if indexs.shape[0] == 1:
                    indexs = indexs[0]
                elif indexs.shape[1] == 1:
                    indexs = indexs[:, 0]
            indexs = np.asarray([str(x).strip() for x in indexs.ravel()])
    elif indexFile.endswith("npy"):
        npy = True
        indexs = np.load(indexFile)
    else:
        npy = False
        raise RuntimeError("index file is not support, we only read the keys of [*.mat, *.npy].")
    labels = scio.loadmat(labelFile)
    if "category" in labels:
        labels = labels["category"]
    elif "LAll" in labels:
        labels = labels["LAll"]
    elif "labels" in labels:
        labels = labels["labels"]
    elif "label" in labels:
        labels = labels["label"]
    else:
        raise RuntimeError("label file is not support, we only read the keys of [caption, tags, YAll].")

    # prompt caption - 支持主提示和患者级上下文
    prompt_caption_path = promptCaptionFile or f"dataset/{dataset_name}/prompt_caption.npz"

    # 检查文件是否存在
    if os.path.exists(prompt_caption_path):
        pc = np.load(prompt_caption_path)
        # ——主提示：ids 优先，其次嵌入
        if "prompt_caption_ids" in pc.files:
            prompt_captions = pc["prompt_caption_ids"]     # (N, 77) int32
            prompt_embed = None
        else:
            prompt_captions = None
            prompt_embed = pc["prompt_caption"] if "prompt_caption" in pc.files else None
        # ——上下文：同样优先 ids
        patient_context_ids = pc["patient_context_ids"] if "patient_context_ids" in pc.files else None
        patient_context_emb = pc["patient_context"] if "patient_context" in pc.files else None
    else:
        print(f"⚠️  未找到prompt_caption文件: {prompt_caption_path}")
        prompt_captions = None
        prompt_embed = None
        patient_context_ids = None
        patient_context_emb = None

    if dataset_name == 'flickr25k':
        img_path = flickr25k_img_path
    elif dataset_name == 'nuswide':
        img_path = nuswide_img_path
    elif dataset_name == 'odir':
        # 使用相对路径，数据集在data/odir目录下
        img_path = f"dataset/{dataset_name}/"
    elif dataset_name == 'mimic-cxr':
        # 使用相对路径，数据集在dataset/mimic-cxr目录下
        img_path = f"dataset/{dataset_name}/"
    elif dataset_name == 'iu-xray':
        # 使用相对路径，数据集在dataset/iu-xray目录下
        img_path = f"dataset/{dataset_name}/"
    else:
        img_path = coco_img_path

    split_indexs, split_captions, split_labels, split_prompt_captions, split_indices = split_data(
        captions,
        indexs,
        labels,
        prompt_captions,
        query_num=query_num,
        train_num=train_num,
        seed=seed,
        return_split_indices=True,
    )
    split_prompt_embed = _split_optional_array(prompt_embed, split_indices)
    split_patient_context_ids = _split_optional_array(patient_context_ids, split_indices)
    split_patient_context_emb = _split_optional_array(patient_context_emb, split_indices)

    query_data = BaseDataset(captions=split_captions[0], indexs=split_indexs[0], labels=split_labels[0],
                             prompt_captions=split_prompt_captions[0], prompt_embed=split_prompt_embed[0],
                             patient_context_ids=split_patient_context_ids[0], patient_context_emb=split_patient_context_emb[0],
                             img_path=img_path, dataset_name=dataset_name, maxWords=maxWords, imageResolution=imageResolution, is_train=False)
    train_data = BaseDataset(captions=split_captions[1], indexs=split_indexs[1], labels=split_labels[1],
                             prompt_captions=split_prompt_captions[1], prompt_embed=split_prompt_embed[1],
                             patient_context_ids=split_patient_context_ids[1], patient_context_emb=split_patient_context_emb[1],
                             img_path=img_path, dataset_name=dataset_name, maxWords=maxWords, imageResolution=imageResolution)
    retrieval_data = BaseDataset(captions=split_captions[2], indexs=split_indexs[2], labels=split_labels[2],
                                 prompt_captions=split_prompt_captions[2], prompt_embed=split_prompt_embed[2],
                                 patient_context_ids=split_patient_context_ids[2], patient_context_emb=split_patient_context_emb[2],
                                 img_path=img_path, dataset_name=dataset_name, maxWords=maxWords, imageResolution=imageResolution, is_train=False)

    return train_data, query_data, retrieval_data
