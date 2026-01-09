"""
BERT vs 传统机器学习模型 - 对比实验
全面对比BERT模型与传统机器学习模型的性能
"""
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
import argparse
import os
from transformers import BertTokenizer
from tqdm import tqdm
import time

from config import Config
from model import SentimentClassifier
from load_data import DataLoader as DataLoaderClass

# 设置绘图样式
plt.style.use('default')
plt.rcParams['axes.unicode_minus'] = False

def evaluate_bert(test_texts, test_labels, device, config):
    """评估BERT模型"""
    print("\n" + "="*70)
    print("评估 BERT 模型")
    print("="*70)

    # 加载模型
    tokenizer = BertTokenizer.from_pretrained(config.model_name)
    model = SentimentClassifier(config.model_name, config.num_classes)

    model_path = config.model_save_path
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"BERT模型未找到: {model_path}")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # 预测
    all_predictions = []
    all_probabilities = []
    batch_size = config.batch_size

    start_time = time.time()

    for i in tqdm(range(0, len(test_texts), batch_size), desc="BERT推理"):
        batch_texts = test_texts[i:i+batch_size]

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

        with torch.no_grad():
            outputs = model(input_ids, attention_mask)
            probs = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(outputs, dim=1)

        all_predictions.extend(predictions.cpu().numpy())
        all_probabilities.extend(probs[:, 1].cpu().numpy())

    pred_time = time.time() - start_time

    # 计算指标
    y_pred = np.array(all_predictions)
    y_prob = np.array(all_probabilities)
    y_true = np.array(test_labels)

    results = {
        'model': 'BERT',
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='binary'),
        'recall': recall_score(y_true, y_pred, average='binary'),
        'f1_score': f1_score(y_true, y_pred, average='binary'),
        'auc_roc': roc_auc_score(y_true, y_prob),
        'pred_time': pred_time,
        'y_pred': y_pred,
        'y_prob': y_prob
    }

    print(f"✓ BERT评估完成 (耗时: {pred_time:.2f}秒)")
    return results

def evaluate_traditional_models(train_texts, train_labels, test_texts, test_labels, max_features=5000):
    """评估传统机器学习模型"""
    print("\n" + "="*70)
    print("评估传统机器学习模型")
    print("="*70)

    # TF-IDF向量化
    print("\n训练TF-IDF向量化器...")
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.8
    )
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)
    y_train = np.array(train_labels)
    y_test = np.array(test_labels)

    print(f"✓ 特征维度: {X_train.shape[1]}")

    # 定义模型
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'SVM (Linear)': SVC(kernel='linear', probability=True, random_state=42),
        'Naive Bayes': MultinomialNB(alpha=1.0)
    }

    results = {}

    for model_name, model in models.items():
        print(f"\n训练 {model_name}...")

        # 训练
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time

        # 预测
        start_time = time.time()
        y_pred = model.predict(X_test)
        pred_time = time.time() - start_time

        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = model.decision_function(X_test)

        # 计算指标
        results[model_name] = {
            'model': model_name,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='binary'),
            'recall': recall_score(y_test, y_pred, average='binary'),
            'f1_score': f1_score(y_test, y_pred, average='binary'),
            'auc_roc': roc_auc_score(y_test, y_prob),
            'train_time': train_time,
            'pred_time': pred_time,
            'y_pred': y_pred,
            'y_prob': y_prob
        }

        print(f"✓ {model_name} 完成 (训练: {train_time:.2f}s, 预测: {pred_time:.2f}s)")

    return results

def plot_comparison(bert_results, traditional_results, output_dir='.'):
    """绘制对比图表"""
    # 准备数据
    all_results = {'BERT': bert_results}
    all_results.update(traditional_results)

    models = list(all_results.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-score', 'AUC-ROC']

    # 1. 指标对比柱状图
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.ravel()

    for idx, (metric, name) in enumerate(zip(metrics, metric_names)):
        values = [all_results[model][metric] for model in models]

        ax = axes[idx]
        bars = ax.bar(range(len(models)), values, color=['#FF6B6B' if m == 'BERT' else '#4ECDC4' for m in models])
        ax.set_xlabel('Model', fontsize=11)
        ax.set_ylabel('Score', fontsize=11)
        ax.set_title(name, fontsize=13, fontweight='bold')
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.set_ylim([0, 1.1])
        ax.grid(axis='y', alpha=0.3)

        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9)

    # 删除多余的子图
    fig.delaxes(axes[5])

    plt.suptitle('BERT vs Traditional ML Models - Performance Comparison', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    comparison_path = os.path.join(output_dir, 'model_comparison.png')
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ 对比图已保存: {comparison_path}")
    plt.close()

    # 2. 时间对比
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 预测时间对比
    pred_times = [all_results[model]['pred_time'] for model in models]
    bars = ax1.bar(range(len(models)), pred_times,
                   color=['#FF6B6B' if m == 'BERT' else '#4ECDC4' for m in models])
    ax1.set_xlabel('Model', fontsize=11)
    ax1.set_ylabel('Time (seconds)', fontsize=11)
    ax1.set_title('Prediction Time Comparison', fontsize=13, fontweight='bold')
    ax1.set_xticks(range(len(models)))
    ax1.set_xticklabels(models, rotation=45, ha='right')
    ax1.grid(axis='y', alpha=0.3)

    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}s',
                ha='center', va='bottom', fontsize=9)

    # F1分数 vs 预测时间散点图
    f1_scores = [all_results[model]['f1_score'] for model in models]
    colors = ['#FF6B6B' if m == 'BERT' else '#4ECDC4' for m in models]

    ax2.scatter(pred_times, f1_scores, c=colors, s=200, alpha=0.6, edgecolors='black')
    for i, model in enumerate(models):
        ax2.annotate(model, (pred_times[i], f1_scores[i]),
                    xytext=(5, 5), textcoords='offset points', fontsize=9)

    ax2.set_xlabel('Prediction Time (seconds)', fontsize=11)
    ax2.set_ylabel('F1-score', fontsize=11)
    ax2.set_title('F1-score vs Prediction Time', fontsize=13, fontweight='bold')
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    time_comparison_path = os.path.join(output_dir, 'time_comparison.png')
    plt.savefig(time_comparison_path, dpi=300, bbox_inches='tight')
    print(f"✓ 时间对比图已保存: {time_comparison_path}")
    plt.close()

def generate_comparison_report(bert_results, traditional_results, output_path):
    """生成详细对比报告"""
    all_results = {'BERT': bert_results}
    all_results.update(traditional_results)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("BERT vs 传统机器学习模型 - 综合对比报告\n")
        f.write("="*80 + "\n\n")

        f.write("【性能指标对比】\n")
        f.write("-"*80 + "\n")
        f.write(f"{'模型':<20} {'准确率':>10} {'精确率':>10} {'召回率':>10} {'F1分数':>10} {'AUC-ROC':>10}\n")
        f.write("-"*80 + "\n")

        for model_name, result in all_results.items():
            f.write(f"{model_name:<20} "
                   f"{result['accuracy']:>10.4f} "
                   f"{result['precision']:>10.4f} "
                   f"{result['recall']:>10.4f} "
                   f"{result['f1_score']:>10.4f} "
                   f"{result['auc_roc']:>10.4f}\n")

        f.write("\n【效率对比】\n")
        f.write("-"*80 + "\n")
        f.write(f"{'模型':<20} {'预测时间':>15}\n")
        f.write("-"*80 + "\n")

        for model_name, result in all_results.items():
            f.write(f"{model_name:<20} {result['pred_time']:>14.2f}秒\n")

        # 分析
        f.write("\n【对比分析】\n")
        f.write("-"*80 + "\n")

        best_f1_model = max(all_results.items(), key=lambda x: x[1]['f1_score'])
        fastest_model = min(all_results.items(), key=lambda x: x[1]['pred_time'])

        f.write(f"\n1. 最佳性能模型: {best_f1_model[0]}\n")
        f.write(f"   F1分数: {best_f1_model[1]['f1_score']:.4f}\n")

        f.write(f"\n2. 最快预测模型: {fastest_model[0]}\n")
        f.write(f"   预测时间: {fastest_model[1]['pred_time']:.2f}秒\n")

        # BERT vs 传统模型的优势
        bert_f1 = bert_results['f1_score']
        best_traditional_f1 = max(traditional_results.values(), key=lambda x: x['f1_score'])['f1_score']
        improvement = (bert_f1 - best_traditional_f1) / best_traditional_f1 * 100

        f.write(f"\n3. BERT相对于最佳传统模型的提升:\n")
        f.write(f"   F1分数提升: {improvement:+.2f}%\n")

        f.write("\n【结论】\n")
        f.write("-"*80 + "\n")
        f.write("• BERT模型: 利用预训练知识，能够更好地理解语义，性能最优\n")
        f.write("• 传统模型: 训练和预测速度快，对计算资源要求低\n")
        f.write("• 实际应用建议: 根据性能需求和计算资源选择合适的模型\n")

        f.write("="*80 + "\n")

    print(f"\n✓ 对比报告已保存: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='BERT vs 传统模型对比实验')
    parser.add_argument('--max-train-samples', type=int, default=10000,
                       help='训练样本数量限制')
    parser.add_argument('--max-test-samples', type=int, default=None,
                       help='测试样本数量限制')
    parser.add_argument('--output-dir', type=str, default='.',
                       help='输出目录')
    args = parser.parse_args()

    print("="*70)
    print("BERT vs 传统机器学习模型 - 对比实验")
    print("="*70 + "\n")

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}\n")

    # 加载数据
    config = Config()
    data_loader = DataLoaderClass(config)

    print("加载数据...")
    train_texts, train_labels = data_loader.load_csv("train.csv")
    val_texts, val_labels = data_loader.load_csv("dev.csv")
    test_texts, test_labels = data_loader.load_csv("test.csv")

    # 合并训练集和验证集
    train_texts = train_texts + val_texts
    train_labels = train_labels + val_labels

    # 限制样本数量
    if len(train_texts) > args.max_train_samples:
        print(f"限制训练样本: {len(train_texts)} -> {args.max_train_samples}")
        train_texts = train_texts[:args.max_train_samples]
        train_labels = train_labels[:args.max_train_samples]

    if args.max_test_samples and len(test_texts) > args.max_test_samples:
        print(f"限制测试样本: {len(test_texts)} -> {args.max_test_samples}")
        test_texts = test_texts[:args.max_test_samples]
        test_labels = test_labels[:args.max_test_samples]

    print(f"\n训练集: {len(train_texts)} 样本")
    print(f"测试集: {len(test_texts)} 样本")

    # 评估BERT模型
    bert_results = evaluate_bert(test_texts, test_labels, device, config)

    # 评估传统模型
    traditional_results = evaluate_traditional_models(
        train_texts, train_labels, test_texts, test_labels
    )

    # 打印结果摘要
    print("\n" + "="*70)
    print("实验结果摘要")
    print("="*70)

    all_results = {'BERT': bert_results}
    all_results.update(traditional_results)

    print(f"\n{'模型':<20} {'准确率':>10} {'F1分数':>10} {'预测时间':>12}")
    print("-"*70)
    for model_name, result in all_results.items():
        print(f"{model_name:<20} "
              f"{result['accuracy']:>10.4f} "
              f"{result['f1_score']:>10.4f} "
              f"{result['pred_time']:>11.2f}s")

    # 生成输出
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # 绘制对比图表
    plot_comparison(bert_results, traditional_results, output_dir)

    # 生成对比报告
    report_path = os.path.join(output_dir, 'model_comparison_report.txt')
    generate_comparison_report(bert_results, traditional_results, report_path)

    # 保存结果到CSV
    results_df = pd.DataFrame([
        {
            '模型': name,
            '准确率': res['accuracy'],
            '精确率': res['precision'],
            '召回率': res['recall'],
            'F1分数': res['f1_score'],
            'AUC-ROC': res['auc_roc'],
            '预测时间(秒)': res['pred_time']
        }
        for name, res in all_results.items()
    ])

    csv_path = os.path.join(output_dir, 'model_comparison_results.csv')
    results_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"✓ 结果已保存: {csv_path}")

    print("\n✅ 对比实验完成！")

if __name__ == "__main__":
    main()
