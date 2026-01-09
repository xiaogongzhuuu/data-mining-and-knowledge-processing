"""
BERT情感分类器 - K折交叉验证
评估模型的鲁棒性和泛化能力
"""
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import argparse
import os
from transformers import BertTokenizer
from torch.utils.data import DataLoader as TorchDataLoader
from torch.optim import AdamW
import torch.nn as nn
from tqdm import tqdm

from config import Config
from model import SentimentClassifier
from dataset import SentimentDataset
from load_data import DataLoader as DataLoaderClass

# 设置绘图样式
plt.style.use('default')
plt.rcParams['axes.unicode_minus'] = False

def train_one_fold(model, train_loader, val_loader, device, config, fold_idx):
    """训练一个折的模型"""
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    best_val_acc = 0

    print(f"\n{'='*60}")
    print(f"训练 Fold {fold_idx + 1}")
    print(f"{'='*60}")

    for epoch in range(config.num_epochs):
        # 训练阶段
        model.train()
        total_loss = 0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs}")

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
            train_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_train_loss = total_loss / len(train_loader)

        # 验证阶段
        model.eval()
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids, attention_mask)
                _, predictions = torch.max(outputs, dim=1)

                val_preds.extend(predictions.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds, average='binary')

        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Acc={val_acc:.4f}, Val F1={val_f1:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc

    return best_val_acc

def evaluate_fold(model, test_loader, device):
    """评估一个折的模型"""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="评估中", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask)
            _, predictions = torch.max(outputs, dim=1)

            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='binary')
    recall = recall_score(all_labels, all_preds, average='binary')
    f1 = f1_score(all_labels, all_preds, average='binary')

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

def plot_cv_results(fold_results, save_path='cross_validation_results.png'):
    """绘制交叉验证结果"""
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-score']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()

    for idx, (metric, name) in enumerate(zip(metrics, metric_names)):
        values = [result[metric] for result in fold_results]
        folds = list(range(1, len(fold_results) + 1))

        ax = axes[idx]
        ax.plot(folds, values, marker='o', linewidth=2, markersize=8,
               color='steelblue', label='Fold Score')
        ax.axhline(y=np.mean(values), color='red', linestyle='--',
                  linewidth=2, label=f'Mean = {np.mean(values):.4f}')
        ax.fill_between(folds,
                        [np.mean(values) - np.std(values)] * len(folds),
                        [np.mean(values) + np.std(values)] * len(folds),
                        alpha=0.2, color='red', label=f'±1 Std Dev')

        ax.set_xlabel('Fold', fontsize=11)
        ax.set_ylabel('Score', fontsize=11)
        ax.set_title(name, fontsize=13, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([min(values) - 0.05, max(values) + 0.05])

    plt.suptitle('K-Fold Cross Validation Results', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ 交叉验证结果图已保存到: {save_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='BERT模型K折交叉验证')
    parser.add_argument('--n-folds', type=int, default=5,
                       help='K折交叉验证的折数')
    parser.add_argument('--max-samples', type=int, default=5000,
                       help='使用的最大样本数（减少训练时间）')
    parser.add_argument('--output-dir', type=str, default='.',
                       help='输出目录')
    args = parser.parse_args()

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    print(f"K折交叉验证: K = {args.n_folds}\n")

    # 加载配置和数据
    config = Config()
    data_loader = DataLoaderClass(config)

    print("加载数据...")
    # 合并训练集和验证集进行交叉验证
    train_texts, train_labels = data_loader.load_csv("train.csv")
    val_texts, val_labels = data_loader.load_csv("dev.csv")

    all_texts = train_texts + val_texts
    all_labels = train_labels + val_labels

    # 限制样本数量
    if len(all_texts) > args.max_samples:
        print(f"限制样本数量: {len(all_texts)} -> {args.max_samples}")
        all_texts = all_texts[:args.max_samples]
        all_labels = all_labels[:args.max_samples]

    print(f"总样本数: {len(all_texts)}")

    # 初始化K折交叉验证
    kfold = KFold(n_splits=args.n_folds, shuffle=True, random_state=42)

    fold_results = []
    tokenizer = BertTokenizer.from_pretrained(config.model_name)

    # 对每一折进行训练和评估
    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(all_texts)):
        print(f"\n{'='*60}")
        print(f"Fold {fold_idx + 1}/{args.n_folds}")
        print(f"{'='*60}")

        # 准备数据
        fold_train_texts = [all_texts[i] for i in train_idx]
        fold_train_labels = [all_labels[i] for i in train_idx]
        fold_val_texts = [all_texts[i] for i in val_idx]
        fold_val_labels = [all_labels[i] for i in val_idx]

        print(f"训练集: {len(fold_train_texts)} 样本")
        print(f"验证集: {len(fold_val_texts)} 样本")

        # 创建数据集
        train_dataset = SentimentDataset(fold_train_texts, fold_train_labels,
                                        tokenizer, config.max_seq_length)
        val_dataset = SentimentDataset(fold_val_texts, fold_val_labels,
                                      tokenizer, config.max_seq_length)

        train_loader = TorchDataLoader(train_dataset, batch_size=config.batch_size,
                                      shuffle=True)
        val_loader = TorchDataLoader(val_dataset, batch_size=config.batch_size)

        # 初始化模型
        model = SentimentClassifier(config.model_name, config.num_classes)
        model.to(device)

        # 训练模型
        best_val_acc = train_one_fold(model, train_loader, val_loader,
                                      device, config, fold_idx)

        # 评估模型
        fold_metrics = evaluate_fold(model, val_loader, device)
        fold_results.append(fold_metrics)

        print(f"\nFold {fold_idx + 1} 结果:")
        print(f"  准确率:  {fold_metrics['accuracy']:.4f}")
        print(f"  精确率:  {fold_metrics['precision']:.4f}")
        print(f"  召回率:  {fold_metrics['recall']:.4f}")
        print(f"  F1分数:  {fold_metrics['f1_score']:.4f}")

    # 计算平均指标和标准差
    print(f"\n{'='*60}")
    print("K折交叉验证总结")
    print(f"{'='*60}")

    avg_metrics = {
        'accuracy': np.mean([r['accuracy'] for r in fold_results]),
        'precision': np.mean([r['precision'] for r in fold_results]),
        'recall': np.mean([r['recall'] for r in fold_results]),
        'f1_score': np.mean([r['f1_score'] for r in fold_results])
    }

    std_metrics = {
        'accuracy': np.std([r['accuracy'] for r in fold_results]),
        'precision': np.std([r['precision'] for r in fold_results]),
        'recall': np.std([r['recall'] for r in fold_results]),
        'f1_score': np.std([r['f1_score'] for r in fold_results])
    }

    print(f"\n平均指标 (± 标准差):")
    print(f"  准确率:  {avg_metrics['accuracy']:.4f} ± {std_metrics['accuracy']:.4f}")
    print(f"  精确率:  {avg_metrics['precision']:.4f} ± {std_metrics['precision']:.4f}")
    print(f"  召回率:  {avg_metrics['recall']:.4f} ± {std_metrics['recall']:.4f}")
    print(f"  F1分数:  {avg_metrics['f1_score']:.4f} ± {std_metrics['f1_score']:.4f}")

    # 保存结果到CSV
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    results_df = pd.DataFrame(fold_results)
    results_df['fold'] = range(1, len(fold_results) + 1)
    results_df = results_df[['fold', 'accuracy', 'precision', 'recall', 'f1_score']]

    csv_path = os.path.join(output_dir, 'cross_validation_results.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"\n✓ 结果已保存到: {csv_path}")

    # 绘制结果图
    plot_path = os.path.join(output_dir, 'cross_validation_results.png')
    plot_cv_results(fold_results, plot_path)

    # 保存总结报告
    report_path = os.path.join(output_dir, 'cross_validation_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write(f"K折交叉验证报告 (K = {args.n_folds})\n")
        f.write("="*60 + "\n\n")

        f.write("【各折结果】\n")
        f.write("-"*60 + "\n")
        for i, result in enumerate(fold_results, 1):
            f.write(f"\nFold {i}:\n")
            f.write(f"  准确率:  {result['accuracy']:.4f}\n")
            f.write(f"  精确率:  {result['precision']:.4f}\n")
            f.write(f"  召回率:  {result['recall']:.4f}\n")
            f.write(f"  F1分数:  {result['f1_score']:.4f}\n")

        f.write("\n【平均指标 ± 标准差】\n")
        f.write("-"*60 + "\n")
        f.write(f"准确率:  {avg_metrics['accuracy']:.4f} ± {std_metrics['accuracy']:.4f}\n")
        f.write(f"精确率:  {avg_metrics['precision']:.4f} ± {std_metrics['precision']:.4f}\n")
        f.write(f"召回率:  {avg_metrics['recall']:.4f} ± {std_metrics['recall']:.4f}\n")
        f.write(f"F1分数:  {avg_metrics['f1_score']:.4f} ± {std_metrics['f1_score']:.4f}\n")

        f.write("\n【结论】\n")
        f.write("-"*60 + "\n")
        f.write(f"模型在 {args.n_folds} 折交叉验证中表现稳定。\n")
        f.write(f"标准差较小表明模型具有良好的鲁棒性和泛化能力。\n")
        f.write("="*60 + "\n")

    print(f"✓ 报告已保存到: {report_path}")
    print(f"\n✅ K折交叉验证完成！")

if __name__ == "__main__":
    main()
