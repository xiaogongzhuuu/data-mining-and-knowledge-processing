import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import BertTokenizer
from config import Config
from dataset import SentimentDataset
from load_data import DataLoader as DataLoaderClass
from model import SentimentClassifier
import torch.nn as nn
import os
from tqdm.auto import tqdm

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def set_hf_mirrors():
    """
    设置Hugging Face镜像，加速模型下载
    """
    # 设置环境变量
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    # 可选的其他镜像
    # os.environ['HF_ENDPOINT'] = 'https://huggingface.tuna.tsinghua.edu.cn'
    # os.environ['HF_ENDPOINT'] = 'https://mirror.sjtu.edu.cn/hugging-face'

    # 设置模型缓存目录（可选）
    os.environ['HF_HOME'] = './hf_cache'

# 设置镜像
set_hf_mirrors()

def evaluate(model, eval_loader, device):
    """
    评估模型性能

    参数:
        model: 模型对象
        eval_loader: 评估数据加载器
        device: 计算设备（CPU/GPU）

    返回:
        Tuple[float, float]: 平均损失和准确率
    """
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        eval_bar = tqdm(eval_loader, desc="Evaluating", leave=False)
        for batch in eval_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask)
            loss = nn.CrossEntropyLoss()(outputs, labels)

            _, predictions = torch.max(outputs, dim=1)
            correct_predictions += torch.sum(predictions == labels)
            total_predictions += len(labels)
            total_loss += loss.item()

            eval_bar.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(eval_loader) if len(eval_loader) > 0 else 0.0
    accuracy = (correct_predictions.double() / total_predictions).item() if total_predictions > 0 else 0.0
    return avg_loss, accuracy

def train(train_texts, train_labels, val_texts=None, val_labels=None):
    """
    训练模型

    参数:
        train_texts (List[str]): 训练文本列表
        train_labels (List[int]): 训练标签列表
        val_texts (List[str], optional): 验证文本列表
        val_labels (List[int], optional): 验证标签列表

    返回:
        model: 训练好的模型
    """
    # 清理 GPU 缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 加载配置
    config = Config()

    # 为了在本机环境上更快完成实验，适当减小训练规模和轮数
    # [已移除] 移除硬编码的 epoch 限制，直接使用 config 配置
    # config.num_epochs = min(getattr(config, "num_epochs", 5), 2)


    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 初始化tokenizer和模型
    tokenizer = BertTokenizer.from_pretrained(config.model_name)
    model = SentimentClassifier(config.model_name, config.num_classes)
    model.to(device)

    # 使用 Config 中定义的样本数量限制
    if len(train_texts) > config.max_train_samples:
        print(f"截断训练集: {len(train_texts)} -> {config.max_train_samples}")
        train_texts = train_texts[:config.max_train_samples]
        train_labels = train_labels[:config.max_train_samples]

    if val_texts is not None and val_labels is not None and len(val_texts) > config.max_val_samples:
        print(f"截断验证集: {len(val_texts)} -> {config.max_val_samples}")
        val_texts = val_texts[:config.max_val_samples]
        val_labels = val_labels[:config.max_val_samples]

    # 准备训练数据
    train_dataset = SentimentDataset(train_texts, train_labels, tokenizer, config.max_seq_length)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

    # 准备验证数据
    if val_texts is not None and val_labels is not None:
        val_dataset = SentimentDataset(val_texts, val_labels, tokenizer, config.max_seq_length)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size)

    # 优化器
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)

    # 训练循环
    best_accuracy = 0
    for epoch in range(config.num_epochs):
        model.train()
        total_loss = 0

        train_bar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{config.num_epochs}")
        for batch in train_bar:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask)
            loss = nn.CrossEntropyLoss()(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            train_bar.set_postfix(loss=loss.item())
        
        avg_train_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch + 1}/{config.num_epochs}')
        print(f'Average training loss: {avg_train_loss:.4f}')
        
        if val_texts is not None and val_labels is not None:
            val_loss, val_accuracy = evaluate(model, val_loader, device)
            print(f'Validation Loss: {val_loss:.4f}')
            print(f'Validation Accuracy: {val_accuracy:.4f}')
            
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                model.save_model(config.model_save_path)
    
    return model
if __name__ == "__main__":
    # 设置Hugging Face镜像
    set_hf_mirrors()
    
    # 加载配置
    config = Config()
    
    # 加载数据
    data_loader = DataLoaderClass(config)
    
    # 分别加载训练集、验证集和测试集
    train_texts, train_labels = data_loader.load_csv("train.csv")
    val_texts, val_labels = data_loader.load_csv("dev.csv")
    test_texts, test_labels = data_loader.load_csv("test.csv")
    
    # 训练模型
    train(train_texts, train_labels, val_texts, val_labels)
