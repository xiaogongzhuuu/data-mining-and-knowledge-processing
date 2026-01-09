"""
BERT情感分类器 - 综合评估脚本
包含所有评估指标：Accuracy, Precision, Recall, F1-score, AUC-ROC, 混淆矩阵
"""
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
import argparse
import os
from transformers import BertTokenizer
from config import Config
from model import SentimentClassifier
from load_data import DataLoader as DataLoaderClass
from tqdm import tqdm

# 设置绘图样式（移除中文字体，使用英文标签）
plt.style.use('default')
plt.rcParams['axes.unicode_minus'] = False

def load_model(device, config):
    """加载训练好的模型"""
    tokenizer = BertTokenizer.from_pretrained(config.model_name)
    model = SentimentClassifier(config.model_name, config.num_classes)

    model_path = config.model_save_path
    if os.path.exists(model_path):
        print(f"✓ 加载模型: {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        raise FileNotFoundError(f"模型文件未找到: {model_path}")

    model.to(device)
    model.eval()
    return model, tokenizer

def predict_with_probabilities(texts, labels, model, tokenizer, device, config, batch_size=32):
    """
    批量预测并返回预测标签和概率

    返回:
        predictions: 预测的类别标签
        probabilities: 预测为正类(1)的概率
        true_labels: 真实标签
    """
    all_predictions = []
    all_probabilities = []
    all_labels = []

    print("执行模型推理...")
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i+batch_size]
        batch_labels = labels[i:i+batch_size]

        # 编码
        encoded = tokenizer.batch_encode_plus(
            batch_texts,
            max_length=config.max_seq_length,
            add_special_tokens=True,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)

        # 预测
        with torch.no_grad():
            outputs = model(input_ids, attention_mask)
            # 使用softmax获取概率
            probs = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(outputs, dim=1)

        all_predictions.extend(predictions.cpu().numpy())
        # 获取正类（类别1）的概率
        all_probabilities.extend(probs[:, 1].cpu().numpy())
        all_labels.extend(batch_labels)

    return np.array(all_predictions), np.array(all_probabilities), np.array(all_labels)

def plot_confusion_matrix(y_true, y_pred, save_path='confusion_matrix.png'):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative (0)', 'Positive (1)'],
                yticklabels=['Negative (0)', 'Positive (1)'])
    plt.title('Confusion Matrix', fontsize=16, pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ 混淆矩阵已保存到: {save_path}")
    plt.close()

def plot_roc_curve(y_true, y_prob, save_path='roc_curve.png'):
    """绘制ROC曲线"""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    auc_score = roc_auc_score(y_true, y_prob)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
             label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve (Receiver Operating Characteristic)', fontsize=16, pad=20)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ ROC曲线已保存到: {save_path}")
    plt.close()

    return auc_score

def generate_evaluation_report(y_true, y_pred, y_prob, save_path='evaluation_report.txt'):
    """生成详细的评估报告"""
    # 计算各项指标
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='binary')
    recall = recall_score(y_true, y_pred, average='binary')
    f1 = f1_score(y_true, y_pred, average='binary')
    auc = roc_auc_score(y_true, y_prob)

    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # 生成分类报告
    class_report = classification_report(y_true, y_pred,
                                        target_names=['Negative (0)', 'Positive (1)'])

    # 写入报告
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("BERT 情感分类器 - 综合评估报告\n")
        f.write("="*60 + "\n\n")

        f.write("【核心评估指标】\n")
        f.write("-"*60 + "\n")
        f.write(f"准确率 (Accuracy):        {accuracy:.4f} ({accuracy*100:.2f}%)\n")
        f.write(f"精确率 (Precision):       {precision:.4f} ({precision*100:.2f}%)\n")
        f.write(f"召回率 (Recall):          {recall:.4f} ({recall*100:.2f}%)\n")
        f.write(f"F1分数 (F1-score):        {f1:.4f}\n")
        f.write(f"AUC-ROC:                  {auc:.4f}\n\n")

        f.write("【混淆矩阵统计】\n")
        f.write("-"*60 + "\n")
        f.write(f"真阴性 (True Negative):   {tn}\n")
        f.write(f"假阳性 (False Positive):  {fp}\n")
        f.write(f"假阴性 (False Negative):  {fn}\n")
        f.write(f"真阳性 (True Positive):   {tp}\n\n")

        f.write("【详细分类报告】\n")
        f.write("-"*60 + "\n")
        f.write(class_report)
        f.write("\n")

        f.write("【指标说明】\n")
        f.write("-"*60 + "\n")
        f.write("• 准确率(Accuracy): 正确预测的样本占总样本的比例\n")
        f.write("• 精确率(Precision): 预测为正类中真正为正类的比例\n")
        f.write("• 召回率(Recall): 真正的正类中被正确预测的比例\n")
        f.write("• F1分数(F1-score): 精确率和召回率的调和平均数\n")
        f.write("• AUC-ROC: ROC曲线下的面积，衡量分类器区分能力\n")
        f.write("="*60 + "\n")

    print(f"✓ 评估报告已保存到: {save_path}")

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc_roc': auc
    }

def main():
    parser = argparse.ArgumentParser(description='BERT模型综合评估')
    parser.add_argument('--test-samples', type=int, default=None,
                       help='测试样本数量限制（默认使用全部）')
    parser.add_argument('--output-dir', type=str, default='.',
                       help='输出目录')
    args = parser.parse_args()

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}\n")

    # 加载配置和数据
    config = Config()
    data_loader = DataLoaderClass(config)

    print("加载测试数据...")
    test_texts, test_labels = data_loader.load_csv("test.csv")

    # 限制样本数量（如果指定）
    if args.test_samples and args.test_samples < len(test_texts):
        print(f"使用 {args.test_samples} 个测试样本")
        test_texts = test_texts[:args.test_samples]
        test_labels = test_labels[:args.test_samples]
    else:
        print(f"使用全部 {len(test_texts)} 个测试样本")

    # 加载模型
    model, tokenizer = load_model(device, config)

    # 预测
    predictions, probabilities, true_labels = predict_with_probabilities(
        test_texts, test_labels, model, tokenizer, device, config
    )

    print("\n" + "="*60)
    print("生成评估结果...")
    print("="*60 + "\n")

    # 生成输出路径
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    confusion_path = os.path.join(output_dir, 'confusion_matrix.png')
    roc_path = os.path.join(output_dir, 'roc_curve.png')
    report_path = os.path.join(output_dir, 'evaluation_report.txt')

    # 绘制混淆矩阵
    plot_confusion_matrix(true_labels, predictions, confusion_path)

    # 绘制ROC曲线
    auc_score = plot_roc_curve(true_labels, probabilities, roc_path)

    # 生成评估报告
    metrics = generate_evaluation_report(true_labels, predictions, probabilities, report_path)

    # 在控制台输出主要指标
    print("\n" + "="*60)
    print("评估结果摘要")
    print("="*60)
    print(f"准确率 (Accuracy):   {metrics['accuracy']:.4f}")
    print(f"精确率 (Precision):  {metrics['precision']:.4f}")
    print(f"召回率 (Recall):     {metrics['recall']:.4f}")
    print(f"F1分数 (F1-score):   {metrics['f1_score']:.4f}")
    print(f"AUC-ROC:             {metrics['auc_roc']:.4f}")
    print("="*60 + "\n")

    print("✅ 综合评估完成！")
    print(f"\n生成的文件:")
    print(f"  • {confusion_path}")
    print(f"  • {roc_path}")
    print(f"  • {report_path}")

if __name__ == "__main__":
    main()
