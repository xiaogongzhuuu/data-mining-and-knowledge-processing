"""
重新生成训练曲线图（使用改进后的绘图代码）
"""

import json
import matplotlib.pyplot as plt
import numpy as np

# 读取训练历史
with open('./logs/training_history.json', 'r') as f:
    history = json.load(f)

print("📊 重新生成训练曲线图...")
print(f"   数据包含 {len(history['train_loss'])} 个 epochs")

# 改进的绘图函数
def plot_training_curves(history, save_path):
    """绘制训练曲线（改进版：更美观的配色和样式）"""
    
    # 设置epochs为整数列表
    num_epochs = len(history['train_loss'])
    epochs = list(range(1, num_epochs + 1))
    
    # 设置更美观的样式
    try:
        plt.style.use('seaborn-v0_8-darkgrid')
    except:
        plt.style.use('default')
    
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
    print(f"✅ Training curves saved to {save_path}")
    plt.close()

# 生成图表
plot_training_curves(history, './outputs/training_curves.png')

print("\n📊 改进内容:")
print("  ✅ 修复：X轴使用整数刻度（Epoch 1, 2, 3...）")
print("  ✅ 改进：使用专业配色（深蓝色+紫红色）")
print("  ✅ 美化：添加数据点标记（圆圈+方块）")
print("  ✅ 优化：增加线条粗细和透明度")
print("  ✅ 增强：Accuracy显示为百分比")
print("  ✅ 标注：绿色星标标记最佳epoch")
print("  ✅ 提升：300 DPI高清输出")
print("\n图表已更新！请查看 outputs/training_curves.png")


