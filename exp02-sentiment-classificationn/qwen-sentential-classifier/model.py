import torch
import torch.nn as nn
from transformers import AutoModel

class SentimentClassifier(nn.Module):
    """
    情感分类器模型
    基于预训练语言模型（如BERT、Qwen等）的情感分类器
    """
    def __init__(self, model_name, num_classes, freeze_base=True):
        """
        初始化模型

        参数:
            model_name (str): 预训练模型名称
            num_classes (int): 分类类别数量
            freeze_base (bool): 是否冻结预训练模型参数
        """
        super(SentimentClassifier, self).__init__()

        # 加载预训练模型
        self.base_model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=False  # Qwen2不需要trust_remote_code
        )

        # 冻结预训练模型的参数，只训练分类层
        if freeze_base:
            for param in self.base_model.parameters():
                param.requires_grad = False

        # Dropout层，防止过拟合
        self.dropout = nn.Dropout(0.3)

        # 分类器层
        self.classifier = nn.Linear(self.base_model.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        """
        前向传播

        参数:
            input_ids: 输入token的ID
            attention_mask: 注意力掩码

        返回:
            logits: 分类logits
        """
        # 通过预训练模型获取输出
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # 获取[CLS]标记的输出（第一个token）
        # 对于BERT: outputs[1]是pooled_output
        # 对于其他模型: 我们取last_hidden_state的第一个token
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            pooled_output = outputs.pooler_output
        else:
            # 如果没有pooler_output，使用last_hidden_state的第一个token
            pooled_output = outputs.last_hidden_state[:, 0, :]

        # 应用dropout
        output = self.dropout(pooled_output)

        # 通过分类器得到logits
        logits = self.classifier(output)

        return logits

    def save_model(self, path):
        """
        保存模型参数

        参数:
            path (str): 保存路径
        """
        import os
        # 确保目录存在
        dir_name = os.path.dirname(path)
        if dir_name:  # 只有当路径包含目录时才创建
            os.makedirs(dir_name, exist_ok=True)
        torch.save(self.state_dict(), path)
        print(f"模型已保存到: {path}")

    def load_model(self, path):
        """
        加载模型参数

        参数:
            path (str): 模型文件路径
        """
        self.load_state_dict(torch.load(path, map_location='cpu'))
        print(f"模型已从 {path} 加载")
