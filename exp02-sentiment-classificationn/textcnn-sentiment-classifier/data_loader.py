"""
数据加载和预处理模块
"""

import re
import csv
import pickle
from collections import Counter
from typing import List, Tuple, Dict
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import config


def clean_text(text: str) -> str:
    """
    文本清洗：去除特殊字符、多余空格等
    """
    # 转小写
    text = text.lower()
    
    # 去除HTML标签
    text = re.sub(r'<[^>]+>', ' ', text)
    
    # 去除URL
    text = re.sub(r'http\S+|www\S+', ' ', text)
    
    # 保留字母、数字、基本标点
    text = re.sub(r'[^a-z0-9\s.!?,\'-]', ' ', text)
    
    # 去除多余空格
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def simple_tokenize(text: str) -> List[str]:
    """
    简单分词：按空格分割
    """
    text = clean_text(text)
    tokens = text.split()
    return tokens


def load_data_from_csv(filepath: str, max_samples: int = None) -> Tuple[List[str], List[int]]:
    """
    从CSV文件加载数据
    格式: label, title, text
    """
    texts = []
    labels = []
    
    print(f"Loading data from {filepath}...")
    
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if max_samples and i >= max_samples:
                break
            
            if len(row) < 2:
                continue
            
            try:
                # 读取标签
                label = int(row[0].strip())
                
                # 合并 title 和 text
                text = ' '.join(r.strip() for r in row[1:] if r.strip())
                
                if not text:
                    continue
                
                # 标签映射: 1(负面)->0, 2(正面)->1
                mapped_label = 0 if label == 1 else 1
                
                texts.append(text)
                labels.append(mapped_label)
                
            except (ValueError, IndexError):
                continue
    
    print(f"  Loaded {len(texts)} samples")
    return texts, labels


class Vocabulary:
    """
    词表类
    """
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.word_freq = Counter()
        
        # 特殊标记
        self.pad_token = config.PADDING_TOKEN
        self.unk_token = config.UNKNOWN_TOKEN
        
        # 初始化特殊标记
        self.word2idx[self.pad_token] = 0
        self.word2idx[self.unk_token] = 1
        self.idx2word[0] = self.pad_token
        self.idx2word[1] = self.unk_token
    
    def build_from_texts(self, texts: List[str], min_freq: int = 2, max_size: int = 50000):
        """
        从文本构建词表
        """
        print("\nBuilding vocabulary...")
        
        # 统计词频
        for text in texts:
            tokens = simple_tokenize(text)
            self.word_freq.update(tokens)
        
        print(f"  Total unique words: {len(self.word_freq)}")
        
        # 按词频排序，保留高频词
        most_common = self.word_freq.most_common(max_size)
        
        # 过滤低频词
        idx = len(self.word2idx)
        for word, freq in most_common:
            if freq >= min_freq:
                if word not in self.word2idx:
                    self.word2idx[word] = idx
                    self.idx2word[idx] = word
                    idx += 1
        
        print(f"  Vocabulary size: {len(self.word2idx)} (min_freq={min_freq})")
        return self
    
    def encode(self, text: str) -> List[int]:
        """
        将文本转换为索引序列
        """
        tokens = simple_tokenize(text)
        indices = [self.word2idx.get(token, self.word2idx[self.unk_token]) 
                   for token in tokens]
        return indices
    
    def decode(self, indices: List[int]) -> str:
        """
        将索引序列转换为文本
        """
        words = [self.idx2word.get(idx, self.unk_token) for idx in indices]
        return ' '.join(words)
    
    def __len__(self):
        return len(self.word2idx)
    
    def save(self, filepath: str):
        """保存词表"""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"Vocabulary saved to {filepath}")
    
    @staticmethod
    def load(filepath: str):
        """加载词表"""
        with open(filepath, 'rb') as f:
            vocab = pickle.load(f)
        print(f"Vocabulary loaded from {filepath}")
        return vocab


class SentimentDataset(Dataset):
    """
    情感分类数据集
    """
    def __init__(self, texts: List[str], labels: List[int], vocab: Vocabulary, max_length: int):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # 编码文本
        indices = self.vocab.encode(text)
        
        # 截断或填充
        if len(indices) > self.max_length:
            indices = indices[:self.max_length]
        else:
            indices = indices + [self.vocab.word2idx[self.vocab.pad_token]] * (self.max_length - len(indices))
        
        return {
            'input_ids': torch.tensor(indices, dtype=torch.long),
            'labels': torch.tensor(label, dtype=torch.long),
            'length': torch.tensor(min(len(self.vocab.encode(text)), self.max_length), dtype=torch.long)
        }


def create_data_loaders(vocab: Vocabulary = None):
    """
    创建训练、验证和测试数据加载器
    """
    # 加载数据（先不限制数量）
    train_texts, train_labels = load_data_from_csv(config.TRAIN_FILE, max_samples=None)
    dev_texts, dev_labels = load_data_from_csv(config.DEV_FILE)
    test_texts, test_labels = load_data_from_csv(config.TEST_FILE)
    
    # 均衡采样训练集
    if config.MAX_TRAIN_SAMPLES and len(train_texts) > config.MAX_TRAIN_SAMPLES:
        if hasattr(config, 'BALANCE_TRAIN_DATA') and config.BALANCE_TRAIN_DATA:
            print(f"\n⚖️  Balanced sampling {config.MAX_TRAIN_SAMPLES} training samples...")
            
            import random
            # 分离正负样本
            neg_indices = [i for i, label in enumerate(train_labels) if label == 0]
            pos_indices = [i for i, label in enumerate(train_labels) if label == 1]
            
            # 每类采样一半
            samples_per_class = config.MAX_TRAIN_SAMPLES // 2
            
            rng = random.Random(42)
            rng.shuffle(neg_indices)
            rng.shuffle(pos_indices)
            
            selected_neg = neg_indices[:samples_per_class]
            selected_pos = pos_indices[:samples_per_class]
            
            # 合并并打乱
            selected_indices = selected_neg + selected_pos
            rng.shuffle(selected_indices)
            
            train_texts = [train_texts[i] for i in selected_indices]
            train_labels = [train_labels[i] for i in selected_indices]
            
            print(f"   ✅ Sampled {len(selected_neg)} negative + {len(selected_pos)} positive = {len(train_texts)} total")
        else:
            # 随机采样（不均衡）
            train_texts = train_texts[:config.MAX_TRAIN_SAMPLES]
            train_labels = train_labels[:config.MAX_TRAIN_SAMPLES]
            print(f"\n🚀 Sampled {config.MAX_TRAIN_SAMPLES} training samples")
    
    # 构建或加载词表
    if vocab is None:
        vocab = Vocabulary()
        vocab.build_from_texts(train_texts, config.MIN_WORD_FREQ, config.MAX_VOCAB_SIZE)
        vocab.save(config.VOCAB_SAVE_PATH)
    
    # 打印数据统计
    print("\n" + "="*60)
    print("Dataset Statistics:")
    print(f"  Train: {len(train_texts)} samples (neg: {train_labels.count(0)}, pos: {train_labels.count(1)})")
    print(f"  Dev:   {len(dev_texts)} samples (neg: {dev_labels.count(0)}, pos: {dev_labels.count(1)})")
    print(f"  Test:  {len(test_texts)} samples (neg: {test_labels.count(0)}, pos: {test_labels.count(1)})")
    print(f"  Vocabulary size: {len(vocab)}")
    print(f"  Max sequence length: {config.MAX_SEQ_LENGTH}")
    print("="*60)
    
    # 创建数据集
    train_dataset = SentimentDataset(train_texts, train_labels, vocab, config.MAX_SEQ_LENGTH)
    dev_dataset = SentimentDataset(dev_texts, dev_labels, vocab, config.MAX_SEQ_LENGTH)
    test_dataset = SentimentDataset(test_texts, test_labels, vocab, config.MAX_SEQ_LENGTH)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    return train_loader, dev_loader, test_loader, vocab


if __name__ == "__main__":
    # 测试数据加载
    train_loader, dev_loader, test_loader, vocab = create_data_loaders()
    
    # 打印一个batch的示例
    for batch in train_loader:
        print("\nSample batch:")
        print(f"  input_ids shape: {batch['input_ids'].shape}")
        print(f"  labels shape: {batch['labels'].shape}")
        print(f"  First sample:")
        print(f"    indices: {batch['input_ids'][0][:20].tolist()}...")
        print(f"    label: {batch['labels'][0].item()}")
        print(f"    decoded: {vocab.decode(batch['input_ids'][0][:20].tolist())}")
        break

