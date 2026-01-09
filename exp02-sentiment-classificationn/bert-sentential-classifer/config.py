import os

# 获取当前文件所在目录
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

class Config:
    """
    模型配置类，包含所有可配置参数
    """
    # 模型参数
    model_name = "bert-base-uncased"  # 预训练模型名称
    # 提示：如果显存依然不足，可以尝试更换为更小的模型，例如 "hfl/rbt3" (3层BERT)
    
    max_seq_length = 64  # [调整] 减小序列长度以节省显存 (原128)
    num_classes = 2  # 分类类别数量，二分类为2
    
    # 训练参数
    batch_size = 8  # [调整] 减小批次大小 (原16)
    learning_rate = 2e-5  # 学习率
    num_epochs = 1  # [调整] 训练轮数
    
    # 数据量限制 (用于低配设备或快速调试)
    max_train_samples = 2000  # [新增] 限制训练样本数
    max_val_samples = 500  # [新增] 限制验证样本数
    
    # 路径配置 (使用绝对路径)
    train_path = os.path.join(_CURRENT_DIR, "train.csv")  # 训练集路径
    dev_path = os.path.join(_CURRENT_DIR, "dev.csv")  # 验证集路径
    test_path = os.path.join(_CURRENT_DIR, "test.csv")  # 测试集路径
    model_save_path = os.path.join(_CURRENT_DIR, "sentiment_model.pth")  # 模型保存路径