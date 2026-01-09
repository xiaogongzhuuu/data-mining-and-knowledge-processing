"""
训练集大小对模型准确率的影响分析
使用不同大小的训练集训练模型，观察最终测试准确率的变化
"""
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import BertTokenizer
import torch.nn as nn
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import argparse
import json

from config import Config
from dataset import SentimentDataset
from load_data import DataLoader as DataLoaderClass
from model import SentimentClassifier

# 设置Hugging Face镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HOME'] = './hf_cache'

def set_seed(seed=42):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def evaluate(model, eval_loader, device):
    """评估模型"""
    model.eval()
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for batch in eval_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask)
            _, predictions = torch.max(outputs, dim=1)
            correct_predictions += torch.sum(predictions == labels)
            total_predictions += len(labels)

    accuracy = (correct_predictions.double() / total_predictions).item()
    return accuracy

def train_with_size(train_texts, train_labels, val_texts, val_labels,
                    config, device, train_size, model_save_path):
    """
    使用指定大小的训练集训练模型

    参数:
        train_size: 训练集大小
    """
    print(f"\n{'='*60}")
    print(f"Training with {train_size} samples")
    print(f"{'='*60}")

    # 采样训练数据
    if train_size < len(train_texts):
        indices = random.sample(range(len(train_texts)), train_size)
        sampled_train_texts = [train_texts[i] for i in indices]
        sampled_train_labels = [train_labels[i] for i in indices]
    else:
        sampled_train_texts = train_texts
        sampled_train_labels = train_labels

    # 初始化
    tokenizer = BertTokenizer.from_pretrained(config.model_name)
    model = SentimentClassifier(config.model_name, config.num_classes)
    model.to(device)

    # 准备数据
    train_dataset = SentimentDataset(sampled_train_texts, sampled_train_labels,
                                     tokenizer, config.max_seq_length)
    val_dataset = SentimentDataset(val_texts, val_labels,
                                   tokenizer, config.max_seq_length)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)

    # 优化器
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)

    # 训练循环
    best_val_accuracy = 0
    train_accuracies = []
    val_accuracies = []

    for epoch in range(config.num_epochs):
        model.train()
        total_loss = 0

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.num_epochs}")
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

        # 评估
        train_acc = evaluate(model, train_loader, device)
        val_acc = evaluate(model, val_loader, device)

        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        print(f"Epoch {epoch + 1}: Train Acc = {train_acc:.4f}, Val Acc = {val_acc:.4f}")

        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            torch.save(model.state_dict(), model_save_path)

    # 清理内存
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        'train_size': train_size,
        'best_val_accuracy': best_val_accuracy,
        'final_train_accuracy': train_accuracies[-1],
        'final_val_accuracy': val_accuracies[-1],
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies
    }

def plot_results(df, history_dict, output_dir):
    """绘制结果图表"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Training Set Size Impact on Model Performance',
                 fontsize=16, fontweight='bold')

    # 1. 训练集大小 vs 准确率
    ax1 = axes[0]
    train_sizes = df['train_size'].values
    best_val_accs = df['best_val_accuracy'].values
    final_val_accs = df['final_val_accuracy'].values

    ax1.plot(train_sizes, best_val_accs, marker='o', linewidth=2,
             markersize=8, label='Best Validation Accuracy')
    ax1.plot(train_sizes, final_val_accs, marker='s', linewidth=2,
             markersize=8, label='Final Validation Accuracy', linestyle='--')
    ax1.set_xlabel('Training Set Size', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title('Validation Accuracy vs Training Set Size', fontsize=13)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. 每个训练集大小的学习曲线
    ax2 = axes[1]
    colors = plt.cm.viridis(np.linspace(0, 1, len(train_sizes)))

    for i, train_size in enumerate(train_sizes):
        if train_size in history_dict:
            val_accs = history_dict[train_size]['val_accuracies']
            epochs = range(1, len(val_accs) + 1)
            ax2.plot(epochs, val_accs, marker='o', linewidth=2,
                     label=f'Size {train_size}', color=colors[i])

    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Validation Accuracy', fontsize=12)
    ax2.set_title('Learning Curves for Different Training Sizes', fontsize=13)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "train_size_analysis.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Plot saved to: {plot_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Analyze training set size impact on model accuracy')
    parser.add_argument('--train-sizes', nargs='+', type=int,
                        default=[500, 1000, 2000, 5000],
                        help='List of training set sizes to test')
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of epochs for each training')
    parser.add_argument('--val-size', type=int, default=1000,
                        help='Validation set size')
    args = parser.parse_args()

    # 设置随机种子
    set_seed(42)

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # 加载配置
    config = Config()
    config.num_epochs = args.epochs  # 使用命令行参数

    # 加载数据
    print("Loading data...")
    data_loader = DataLoaderClass(config)
    current_dir = os.path.dirname(os.path.abspath(__file__))

    train_texts, train_labels = data_loader.load_csv(os.path.join(current_dir, "train.csv"))
    val_texts, val_labels = data_loader.load_csv(os.path.join(current_dir, "dev.csv"))

    # 限制验证集大小
    if len(val_texts) > args.val_size:
        indices = random.sample(range(len(val_texts)), args.val_size)
        val_texts = [val_texts[i] for i in indices]
        val_labels = [val_labels[i] for i in indices]

    print(f"Total training samples available: {len(train_texts)}")
    print(f"Validation samples: {len(val_texts)}")

    # 运行实验
    results = []
    history_dict = {}

    for train_size in args.train_sizes:
        if train_size > len(train_texts):
            print(f"\nWarning: Requested train_size {train_size} exceeds available data {len(train_texts)}")
            train_size = len(train_texts)

        model_save_path = os.path.join(current_dir, f"model_trainsize_{train_size}.pth")

        result = train_with_size(
            train_texts, train_labels, val_texts, val_labels,
            config, device, train_size, model_save_path
        )

        results.append(result)
        history_dict[train_size] = result

        print(f"\nResults for train_size={train_size}:")
        print(f"  Best Val Accuracy:  {result['best_val_accuracy']:.4f}")
        print(f"  Final Val Accuracy: {result['final_val_accuracy']:.4f}")

    # 保存结果
    df_results = pd.DataFrame([
        {
            'train_size': r['train_size'],
            'best_val_accuracy': r['best_val_accuracy'],
            'final_val_accuracy': r['final_val_accuracy'],
            'final_train_accuracy': r['final_train_accuracy']
        }
        for r in results
    ])

    csv_path = os.path.join(current_dir, "train_size_results.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"\n✓ Results saved to: {csv_path}")

    # 保存详细历史
    history_path = os.path.join(current_dir, "train_size_history.json")
    with open(history_path, 'w') as f:
        json.dump(history_dict, f, indent=2)
    print(f"✓ Training history saved to: {history_path}")

    # 绘制图表
    plot_results(df_results, history_dict, current_dir)

    # 打印总结
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(df_results.to_string(index=False))

    print("\n✓ Experiment completed!")

if __name__ == "__main__":
    main()