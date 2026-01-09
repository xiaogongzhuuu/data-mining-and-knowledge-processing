"""
训练脚本：TextCNN 情感分类
"""

import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt

import config
from data_loader import create_data_loaders, Vocabulary
from model import create_model


def set_seed(seed):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model, dataloader, device, criterion):
    """
    评估模型
    
    Returns:
        metrics: 包含 loss, acc, f1, precision, recall 的字典
    """
    model.eval()
    
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            # 前向传播
            logits = model(input_ids)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            
            # 预测
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 计算指标
    avg_loss = total_loss / len(dataloader)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='binary')
    precision = precision_score(all_labels, all_preds, average='binary', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='binary', zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    
    metrics = {
        'loss': avg_loss,
        'acc': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'confusion_matrix': cm.tolist()
    }
    
    return metrics


def train_epoch(model, dataloader, optimizer, criterion, device, epoch, num_epochs):
    """训练一个epoch"""
    model.train()
    
    total_loss = 0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
    
    for batch in pbar:
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        
        # 前向传播
        optimizer.zero_grad()
        logits = model(input_ids)
        loss = criterion(logits, labels)
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪（防止梯度爆炸）
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        
        # 预测
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        # 更新进度条
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    # 计算训练集指标
    avg_loss = total_loss / len(dataloader)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='binary')
    
    return avg_loss, acc, f1


def plot_training_curves(history, save_path):
    """绘制训练曲线（改进版：更美观的配色和样式）"""
    import numpy as np
    
    # 设置epochs为整数列表
    num_epochs = len(history['train_loss'])
    epochs = list(range(1, num_epochs + 1))
    
    # 设置更美观的样式
    plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')
    
    # 创建图形，使用更大的尺寸和更好的布局
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('TextCNN Training Curves', fontsize=16, fontweight='bold', y=1.02)
    
    # 定义更美观的颜色（使用专业配色）
    train_color = '#2E86AB'  # 深蓝色
    dev_color = '#A23B72'    # 紫红色
    
    # ============ Loss曲线 ============
    axes[0].plot(epochs, history['train_loss'], 
                color=train_color, linewidth=2.5, marker='o', markersize=6,
                label='Training', alpha=0.9)
    axes[0].plot(epochs, history['dev_loss'], 
                color=dev_color, linewidth=2.5, marker='s', markersize=6,
                label='Validation', alpha=0.9)
    axes[0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Loss', fontsize=12, fontweight='bold')
    axes[0].set_title('Loss Curve', fontsize=13, fontweight='bold', pad=10)
    axes[0].legend(loc='best', fontsize=10, framealpha=0.9)
    axes[0].grid(True, alpha=0.3, linestyle='--')
    axes[0].set_xticks(epochs)  # 设置为整数刻度
    
    # ============ Accuracy曲线 ============
    # 转换为百分比显示
    train_acc_pct = [x * 100 for x in history['train_acc']]
    dev_acc_pct = [x * 100 for x in history['dev_acc']]
    
    axes[1].plot(epochs, train_acc_pct, 
                color=train_color, linewidth=2.5, marker='o', markersize=6,
                label='Training', alpha=0.9)
    axes[1].plot(epochs, dev_acc_pct, 
                color=dev_color, linewidth=2.5, marker='s', markersize=6,
                label='Validation', alpha=0.9)
    axes[1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    axes[1].set_title('Accuracy Curve', fontsize=13, fontweight='bold', pad=10)
    axes[1].legend(loc='best', fontsize=10, framealpha=0.9)
    axes[1].grid(True, alpha=0.3, linestyle='--')
    axes[1].set_xticks(epochs)  # 设置为整数刻度
    axes[1].set_ylim([70, 100])  # 设置y轴范围使曲线更清晰
    
    # ============ F1 Score曲线 ============
    axes[2].plot(epochs, history['train_f1'], 
                color=train_color, linewidth=2.5, marker='o', markersize=6,
                label='Training', alpha=0.9)
    axes[2].plot(epochs, history['dev_f1'], 
                color=dev_color, linewidth=2.5, marker='s', markersize=6,
                label='Validation', alpha=0.9)
    axes[2].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[2].set_ylabel('F1 Score', fontsize=12, fontweight='bold')
    axes[2].set_title('F1 Score Curve', fontsize=13, fontweight='bold', pad=10)
    axes[2].legend(loc='best', fontsize=10, framealpha=0.9)
    axes[2].grid(True, alpha=0.3, linestyle='--')
    axes[2].set_xticks(epochs)  # 设置为整数刻度
    axes[2].set_ylim([0.7, 1.0])  # 设置y轴范围使曲线更清晰
    
    # 标记最佳验证集性能点
    best_epoch = np.argmax(history['dev_f1']) + 1
    best_f1 = max(history['dev_f1'])
    axes[2].axvline(x=best_epoch, color='green', linestyle=':', linewidth=2, alpha=0.6, label=f'Best (Epoch {best_epoch})')
    axes[2].scatter([best_epoch], [best_f1], color='green', s=150, zorder=5, marker='*', edgecolors='darkgreen', linewidths=2)
    axes[2].legend(loc='best', fontsize=10, framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"📊 Training curves saved to {save_path}")
    plt.close()


def main():
    print("="*80)
    print("TextCNN Sentiment Classification Training")
    print("="*80)
    
    # 设置随机种子
    set_seed(config.SEED)
    
    # 设置设备
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"\n📱 Device: {device}")
    
    # 创建数据加载器
    print("\n📂 Loading data...")
    train_loader, dev_loader, test_loader, vocab = create_data_loaders()
    
    # 创建模型
    print("\n🔨 Creating model...")
    model = create_model(vocab_size=len(vocab))
    model = model.to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    
    print(f"📊 Optimizer: Adam (lr={config.LEARNING_RATE}, weight_decay={config.WEIGHT_DECAY})")
    print(f"📊 Loss function: CrossEntropyLoss")
    
    # 训练历史
    history = {
        'train_loss': [],
        'train_acc': [],
        'train_f1': [],
        'dev_loss': [],
        'dev_acc': [],
        'dev_f1': []
    }
    
    # 早停相关
    best_dev_f1 = 0.0
    patience_counter = 0
    
    print("\n" + "="*80)
    print("🏋️  Training Started")
    print("="*80)
    
    for epoch in range(config.NUM_EPOCHS):
        # 训练
        train_loss, train_acc, train_f1 = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch, config.NUM_EPOCHS
        )
        
        # 验证
        print(f"\n📊 Evaluating on dev set...")
        dev_metrics = evaluate(model, dev_loader, device, criterion)
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_f1'].append(train_f1)
        history['dev_loss'].append(dev_metrics['loss'])
        history['dev_acc'].append(dev_metrics['acc'])
        history['dev_f1'].append(dev_metrics['f1'])
        
        # 打印结果
        print(f"\n{'='*80}")
        print(f"Epoch {epoch+1}/{config.NUM_EPOCHS} Results:")
        print(f"  Train | Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | F1: {train_f1:.4f}")
        print(f"  Dev   | Loss: {dev_metrics['loss']:.4f} | Acc: {dev_metrics['acc']:.4f} | F1: {dev_metrics['f1']:.4f}")
        print(f"        | Precision: {dev_metrics['precision']:.4f} | Recall: {dev_metrics['recall']:.4f}")
        
        # 打印混淆矩阵
        cm = dev_metrics['confusion_matrix']
        print(f"\n  Dev Confusion Matrix:")
        print(f"    [[TN={cm[0][0]}, FP={cm[0][1]}],")
        print(f"     [FN={cm[1][0]}, TP={cm[1][1]}]]")
        print(f"{'='*80}")
        
        # 保存最佳模型
        if dev_metrics['f1'] > best_dev_f1:
            best_dev_f1 = dev_metrics['f1']
            patience_counter = 0
            
            # 保存模型
            # 只保存配置的基本值（避免 pickle 错误）
            config_dict = {
                'embedding_dim': config.EMBEDDING_DIM,
                'num_filters': config.NUM_FILTERS,
                'filter_sizes': config.FILTER_SIZES,
                'num_classes': config.NUM_CLASSES,
                'dropout_rate': config.DROPOUT_RATE,
                'max_seq_length': config.MAX_SEQ_LENGTH,
                'vocab_size': len(vocab)
            }
            
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'dev_f1': dev_metrics['f1'],
                'dev_acc': dev_metrics['acc'],
                'config': config_dict
            }, config.MODEL_SAVE_PATH)
            
            print(f"\n💾 Best model saved! (F1: {best_dev_f1:.4f})")
        else:
            patience_counter += 1
            print(f"\n⏳ Patience: {patience_counter}/{config.PATIENCE}")
        
        # 早停检查
        if config.EARLY_STOPPING and patience_counter >= config.PATIENCE:
            print(f"\n⚠️  Early stopping triggered (no improvement for {config.PATIENCE} epochs)")
            break
    
    # 保存训练历史
    history_path = os.path.join(config.LOG_DIR, "training_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\n📝 Training history saved to {history_path}")
    
    # 绘制训练曲线
    curve_path = os.path.join(config.OUTPUT_DIR, "training_curves.png")
    plot_training_curves(history, curve_path)
    
    # 加载最佳模型并在测试集上评估
    print("\n" + "="*80)
    print("🧪 Evaluating on Test Set")
    print("="*80)
    
    checkpoint = torch.load(config.MODEL_SAVE_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_metrics = evaluate(model, test_loader, device, criterion)
    
    print(f"\n📊 Test Set Results:")
    print(f"  Loss: {test_metrics['loss']:.4f}")
    print(f"  Accuracy: {test_metrics['acc']:.4f}")
    print(f"  F1 Score: {test_metrics['f1']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall: {test_metrics['recall']:.4f}")
    
    cm = test_metrics['confusion_matrix']
    print(f"\n  Test Confusion Matrix:")
    print(f"    [[TN={cm[0][0]}, FP={cm[0][1]}],")
    print(f"     [FN={cm[1][0]}, TP={cm[1][1]}]]")
    
    # 保存测试结果
    test_results_path = os.path.join(config.OUTPUT_DIR, "test_results.json")
    with open(test_results_path, 'w') as f:
        json.dump(test_metrics, f, indent=2)
    print(f"\n📝 Test results saved to {test_results_path}")
    
    print("\n" + "="*80)
    print("✅ Training Completed!")
    print(f"📊 Best Dev F1: {best_dev_f1:.4f}")
    print(f"📊 Test F1: {test_metrics['f1']:.4f}")
    print(f"💾 Model saved to: {config.MODEL_SAVE_PATH}")
    print("="*80)


if __name__ == "__main__":
    main()

