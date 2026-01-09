"""
训练轮数（Epochs）对模型准确率的影响分析
训练模型多个epochs，记录每个epoch后的验证准确率，观察学习曲线
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
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for batch in eval_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask)
            loss = nn.CrossEntropyLoss()(outputs, labels)

            _, predictions = torch.max(outputs, dim=1)
            correct_predictions += torch.sum(predictions == labels)
            total_predictions += len(labels)
            total_loss += loss.item()

    avg_loss = total_loss / len(eval_loader)
    accuracy = (correct_predictions.double() / total_predictions).item()
    return avg_loss, accuracy

def train_multiple_epochs(train_texts, train_labels, val_texts, val_labels,
                          test_texts, test_labels, config, device, max_epochs,
                          checkpoint_epochs=None):
    """
    训练模型多个epochs，记录每个epoch的性能

    参数:
        max_epochs: 最大训练轮数
        checkpoint_epochs: 保存模型的epoch列表（如[1,3,5,10]）
    """
    print(f"\n{'='*60}")
    print(f"Training for {max_epochs} epochs")
    print(f"{'='*60}")

    # 初始化
    tokenizer = BertTokenizer.from_pretrained(config.model_name)
    model = SentimentClassifier(config.model_name, config.num_classes)
    model.to(device)

    # 准备数据
    train_dataset = SentimentDataset(train_texts, train_labels,
                                     tokenizer, config.max_seq_length)
    val_dataset = SentimentDataset(val_texts, val_labels,
                                   tokenizer, config.max_seq_length)
    test_dataset = SentimentDataset(test_texts, test_labels,
                                    tokenizer, config.max_seq_length)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size)

    # 优化器
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)

    # 记录历史
    history = {
        'epoch': [],
        'train_loss': [],
        'train_accuracy': [],
        'val_loss': [],
        'val_accuracy': [],
        'test_loss': [],
        'test_accuracy': []
    }

    best_val_accuracy = 0
    best_epoch = 0

    # 训练循环
    for epoch in range(max_epochs):
        model.train()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{max_epochs}")
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
            _, predictions = torch.max(outputs, dim=1)
            correct_predictions += torch.sum(predictions == labels)
            total_predictions += len(labels)

            train_bar.set_postfix(loss=loss.item())

        # 计算训练集指标
        avg_train_loss = total_loss / len(train_loader)
        train_accuracy = (correct_predictions.double() / total_predictions).item()

        # 评估验证集和测试集
        val_loss, val_accuracy = evaluate(model, val_loader, device)
        test_loss, test_accuracy = evaluate(model, test_loader, device)

        # 记录历史
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(avg_train_loss)
        history['train_accuracy'].append(train_accuracy)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_accuracy)
        history['test_loss'].append(test_loss)
        history['test_accuracy'].append(test_accuracy)

        print(f"Epoch {epoch + 1}/{max_epochs}:")
        print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}, Val Acc:   {val_accuracy:.4f}")
        print(f"  Test Loss:  {test_loss:.4f}, Test Acc:  {test_accuracy:.4f}")

        # 保存最佳模型
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_epoch = epoch + 1
            current_dir = os.path.dirname(os.path.abspath(__file__))
            best_model_path = os.path.join(current_dir, "best_epoch_model.pth")
            torch.save(model.state_dict(), best_model_path)

        # 保存检查点
        if checkpoint_epochs and (epoch + 1) in checkpoint_epochs:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            checkpoint_path = os.path.join(current_dir, f"model_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  ✓ Checkpoint saved: {checkpoint_path}")

    print(f"\nBest validation accuracy: {best_val_accuracy:.4f} at epoch {best_epoch}")

    return history, best_val_accuracy, best_epoch

def plot_results(history, output_path):
    """绘制学习曲线"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Training Progress: Epochs Impact on Model Performance',
                 fontsize=16, fontweight='bold')

    epochs = history['epoch']

    # 1. 准确率曲线
    ax1 = axes[0]
    ax1.plot(epochs, history['train_accuracy'], marker='o', linewidth=2,
             markersize=6, label='Train Accuracy')
    ax1.plot(epochs, history['val_accuracy'], marker='s', linewidth=2,
             markersize=6, label='Validation Accuracy')
    ax1.plot(epochs, history['test_accuracy'], marker='^', linewidth=2,
             markersize=6, label='Test Accuracy')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title('Accuracy vs Epochs', fontsize=13)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. 损失曲线
    ax2 = axes[1]
    ax2.plot(epochs, history['train_loss'], marker='o', linewidth=2,
             markersize=6, label='Train Loss')
    ax2.plot(epochs, history['val_loss'], marker='s', linewidth=2,
             markersize=6, label='Validation Loss')
    ax2.plot(epochs, history['test_loss'], marker='^', linewidth=2,
             markersize=6, label='Test Loss')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_title('Loss vs Epochs', fontsize=13)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Plot saved to: {output_path}")
    plt.close()

def plot_overfitting_analysis(history, output_path):
    """绘制过拟合分析图"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Overfitting Analysis',
                 fontsize=16, fontweight='bold')

    epochs = history['epoch']

    # 1. 训练集 vs 验证集准确率差距
    ax1 = axes[0]
    gap = np.array(history['train_accuracy']) - np.array(history['val_accuracy'])
    ax1.plot(epochs, gap, marker='o', linewidth=2, markersize=6, color='red')
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Train Acc - Val Acc', fontsize=12)
    ax1.set_title('Training-Validation Accuracy Gap', fontsize=13)
    ax1.grid(True, alpha=0.3)

    # 2. 验证集与测试集准确率对比
    ax2 = axes[1]
    ax2.plot(epochs, history['val_accuracy'], marker='s', linewidth=2,
             markersize=6, label='Validation Accuracy')
    ax2.plot(epochs, history['test_accuracy'], marker='^', linewidth=2,
             markersize=6, label='Test Accuracy')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Validation vs Test Accuracy', fontsize=13)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Overfitting analysis plot saved to: {output_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Analyze training epochs impact on model accuracy')
    parser.add_argument('--max-epochs', type=int, default=10,
                        help='Maximum number of epochs to train')
    parser.add_argument('--train-size', type=int, default=5000,
                        help='Training set size')
    parser.add_argument('--val-size', type=int, default=1000,
                        help='Validation set size')
    parser.add_argument('--test-size', type=int, default=1000,
                        help='Test set size')
    parser.add_argument('--checkpoint-epochs', nargs='+', type=int,
                        default=[1, 3, 5, 10],
                        help='Epochs to save model checkpoints')
    args = parser.parse_args()

    # 设置随机种子
    set_seed(42)

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # 加载配置
    config = Config()

    # 加载数据
    print("Loading data...")
    data_loader = DataLoaderClass(config)
    current_dir = os.path.dirname(os.path.abspath(__file__))

    train_texts, train_labels = data_loader.load_csv(os.path.join(current_dir, "train.csv"))
    val_texts, val_labels = data_loader.load_csv(os.path.join(current_dir, "dev.csv"))
    test_texts, test_labels = data_loader.load_csv(os.path.join(current_dir, "test.csv"))

    # 采样数据
    if len(train_texts) > args.train_size:
        indices = random.sample(range(len(train_texts)), args.train_size)
        train_texts = [train_texts[i] for i in indices]
        train_labels = [train_labels[i] for i in indices]

    if len(val_texts) > args.val_size:
        indices = random.sample(range(len(val_texts)), args.val_size)
        val_texts = [val_texts[i] for i in indices]
        val_labels = [val_labels[i] for i in indices]

    if len(test_texts) > args.test_size:
        indices = random.sample(range(len(test_texts)), args.test_size)
        test_texts = [test_texts[i] for i in indices]
        test_labels = [test_labels[i] for i in indices]

    print(f"Training samples: {len(train_texts)}")
    print(f"Validation samples: {len(val_texts)}")
    print(f"Test samples: {len(test_texts)}")

    # 运行训练
    history, best_val_acc, best_epoch = train_multiple_epochs(
        train_texts, train_labels, val_texts, val_labels,
        test_texts, test_labels, config, device,
        max_epochs=args.max_epochs,
        checkpoint_epochs=args.checkpoint_epochs
    )

    # 保存结果
    df_history = pd.DataFrame(history)
    csv_path = os.path.join(current_dir, "epoch_analysis_results.csv")
    df_history.to_csv(csv_path, index=False)
    print(f"\n✓ Results saved to: {csv_path}")

    # 保存JSON格式
    json_path = os.path.join(current_dir, "epoch_analysis_history.json")
    with open(json_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"✓ Training history saved to: {json_path}")

    # 绘制学习曲线
    plot_path = os.path.join(current_dir, "epoch_analysis_learning_curves.png")
    plot_results(history, plot_path)

    # 绘制过拟合分析图
    overfitting_path = os.path.join(current_dir, "epoch_analysis_overfitting.png")
    plot_overfitting_analysis(history, overfitting_path)

    # 打印总结
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(df_history.to_string(index=False))
    print(f"\nBest Validation Accuracy: {best_val_acc:.4f} at Epoch {best_epoch}")

    # 分析过拟合
    final_train_acc = history['train_accuracy'][-1]
    final_val_acc = history['val_accuracy'][-1]
    final_test_acc = history['test_accuracy'][-1]
    overfitting_gap = final_train_acc - final_val_acc

    print(f"\nFinal Performance:")
    print(f"  Train Accuracy: {final_train_acc:.4f}")
    print(f"  Val Accuracy:   {final_val_acc:.4f}")
    print(f"  Test Accuracy:  {final_test_acc:.4f}")
    print(f"  Overfitting Gap: {overfitting_gap:.4f}")

    if overfitting_gap > 0.1:
        print("\n⚠️  Warning: Significant overfitting detected (gap > 0.1)")
    elif overfitting_gap > 0.05:
        print("\n⚠️  Caution: Moderate overfitting (gap > 0.05)")
    else:
        print("\n✓ Overfitting is minimal (gap <= 0.05)")

    print("\n✓ Experiment completed!")

if __name__ == "__main__":
    main()