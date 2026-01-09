class Config:
    """
    模型配置类，包含所有可配置参数
    """
    # 模型参数
    model_name = "Qwen/Qwen2-0.5B"  # Qwen2-0.5B 更稳定的轻量级模型
    max_seq_length = 256  # 降低序列长度
    num_classes = 2  # 分类类别数量，二分类为2

    # 训练参数
    batch_size = 16  # 增加batch size
    learning_rate = 1e-5  # 学习率
    num_epochs = 3  # 训练轮数

    # 路径配置
    train_path = "train.csv"  # 训练集路径
    dev_path = "dev.csv"  # 验证集路径
    test_path = "test.csv"  # 测试集路径
    model_save_path = "saved_models/sentiment_model.pth"  # 模型保存路径 