"""
TextCNN 模型定义
基于论文: Convolutional Neural Networks for Sentence Classification (Kim, 2014)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import config


class TextCNN(nn.Module):
    """
    TextCNN 模型
    
    架构:
    1. Embedding层: 将词索引转换为词向量
    2. 卷积层: 使用多个不同大小的卷积核提取n-gram特征
    3. 池化层: 对每个特征图进行max-pooling
    4. 全连接层: 分类
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 300,
        num_filters: int = 100,
        filter_sizes: list = [3, 4, 5],
        num_classes: int = 2,
        dropout_rate: float = 0.5,
        pretrained_embeddings: np.ndarray = None,
        freeze_embeddings: bool = False
    ):
        """
        初始化TextCNN模型
        
        Args:
            vocab_size: 词表大小
            embedding_dim: 词向量维度
            num_filters: 每个卷积核的数量
            filter_sizes: 卷积核大小列表（窗口大小）
            num_classes: 分类数量
            dropout_rate: Dropout比率
            pretrained_embeddings: 预训练词向量 (vocab_size, embedding_dim)
            freeze_embeddings: 是否冻结embedding层
        """
        super(TextCNN, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_filters = num_filters
        self.filter_sizes = filter_sizes
        self.num_classes = num_classes
        
        # 1. Embedding层
        if pretrained_embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(
                torch.FloatTensor(pretrained_embeddings),
                freeze=freeze_embeddings,
                padding_idx=0
            )
            print(f"✓ Loaded pretrained embeddings (freeze={freeze_embeddings})")
        else:
            self.embedding = nn.Embedding(
                vocab_size,
                embedding_dim,
                padding_idx=0
            )
            # Xavier初始化
            nn.init.xavier_uniform_(self.embedding.weight)
            print(f"✓ Initialized random embeddings")
        
        # 2. 卷积层
        # 每个卷积核大小对应一个卷积层
        self.convs = nn.ModuleList([
            nn.Conv2d(
                in_channels=1,              # 输入通道数（单通道，因为是单词向量）
                out_channels=num_filters,   # 输出通道数（卷积核数量）
                kernel_size=(fs, embedding_dim)  # 卷积核大小 (filter_size, embedding_dim)
            )
            for fs in filter_sizes
        ])
        
        # 3. Dropout层
        self.dropout = nn.Dropout(dropout_rate)
        
        # 4. 全连接层
        # 输入维度 = 卷积核数量 × 卷积核大小种类数
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)
        
        print(f"✓ TextCNN initialized:")
        print(f"    Vocab size: {vocab_size}")
        print(f"    Embedding dim: {embedding_dim}")
        print(f"    Filter sizes: {filter_sizes}")
        print(f"    Num filters per size: {num_filters}")
        print(f"    Total feature dim: {num_filters * len(filter_sizes)}")
        print(f"    Dropout: {dropout_rate}")
    
    def forward(self, input_ids):
        """
        前向传播
        
        Args:
            input_ids: (batch_size, seq_length) 词索引序列
        
        Returns:
            logits: (batch_size, num_classes) 分类logits
        """
        # 1. Embedding
        # (batch_size, seq_length) -> (batch_size, seq_length, embedding_dim)
        embedded = self.embedding(input_ids)
        
        # 2. 增加通道维度用于卷积
        # (batch_size, seq_length, embedding_dim) -> (batch_size, 1, seq_length, embedding_dim)
        embedded = embedded.unsqueeze(1)
        
        # 3. 卷积 + ReLU + Max Pooling
        conv_outputs = []
        for conv in self.convs:
            # 卷积: (batch_size, 1, seq_length, embedding_dim) 
            #    -> (batch_size, num_filters, seq_length - filter_size + 1, 1)
            conv_out = F.relu(conv(embedded))
            
            # 去掉最后一维: (batch_size, num_filters, seq_length - filter_size + 1, 1)
            #            -> (batch_size, num_filters, seq_length - filter_size + 1)
            conv_out = conv_out.squeeze(3)
            
            # Max pooling over time: (batch_size, num_filters, seq_length - filter_size + 1)
            #                      -> (batch_size, num_filters)
            pooled = F.max_pool1d(conv_out, conv_out.size(2))
            pooled = pooled.squeeze(2)
            
            conv_outputs.append(pooled)
        
        # 4. 拼接所有卷积核的输出
        # [(batch_size, num_filters)] * len(filter_sizes) -> (batch_size, num_filters * len(filter_sizes))
        concatenated = torch.cat(conv_outputs, dim=1)
        
        # 5. Dropout
        dropped = self.dropout(concatenated)
        
        # 6. 全连接层
        # (batch_size, num_filters * len(filter_sizes)) -> (batch_size, num_classes)
        logits = self.fc(dropped)
        
        return logits
    
    def get_num_params(self):
        """获取模型参数数量"""
        return sum(p.numel() for p in self.parameters())
    
    def get_trainable_params(self):
        """获取可训练参数数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(vocab_size: int, pretrained_embeddings: np.ndarray = None):
    """
    创建TextCNN模型
    
    Args:
        vocab_size: 词表大小
        pretrained_embeddings: 预训练词向量 (可选)
    
    Returns:
        model: TextCNN模型
    """
    model = TextCNN(
        vocab_size=vocab_size,
        embedding_dim=config.EMBEDDING_DIM,
        num_filters=config.NUM_FILTERS,
        filter_sizes=config.FILTER_SIZES,
        num_classes=config.NUM_CLASSES,
        dropout_rate=config.DROPOUT_RATE,
        pretrained_embeddings=pretrained_embeddings,
        freeze_embeddings=config.FREEZE_EMBEDDING
    )
    
    print(f"\n{'='*60}")
    print(f"Model Summary:")
    print(f"  Total parameters: {model.get_num_params():,}")
    print(f"  Trainable parameters: {model.get_trainable_params():,}")
    print(f"{'='*60}\n")
    
    return model


if __name__ == "__main__":
    # 测试模型
    print("Testing TextCNN model...\n")
    
    # 创建模型
    vocab_size = 10000
    model = create_model(vocab_size)
    
    # 测试前向传播
    batch_size = 4
    seq_length = 128
    
    # 随机输入
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
    
    # 前向传播
    logits = model(input_ids)
    
    print(f"Input shape: {input_ids.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Output (logits): {logits}")
    
    # 计算预测
    preds = torch.argmax(logits, dim=1)
    print(f"Predictions: {preds}")

