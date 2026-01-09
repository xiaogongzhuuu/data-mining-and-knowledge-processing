"""
测试集抽样大小对准确率估计稳定性的影响
对于不同的sample_size，重复随机抽样，观察准确率的均值和波动
"""
import torch
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from transformers import BertTokenizer
from config import Config
from model import SentimentClassifier
from load_data import DataLoader as DataLoaderClass
from tqdm import tqdm
import argparse

def load_trained_model(device, config):
    """加载训练好的模型"""
    model = SentimentClassifier(config.model_name, config.num_classes)
    model_path = config.model_save_path

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        raise FileNotFoundError(f"Trained model not found at {model_path}")

    model.to(device)
    model.eval()
    return model

def predict_batch(texts, model, tokenizer, device, config):
    """批量预测"""
    encoded = tokenizer.batch_encode_plus(
        texts,
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
        _, predictions = torch.max(outputs, dim=1)

    return predictions.cpu().numpy()

def evaluate_sample(model, texts, labels, tokenizer, device, config):
    """评估模型在给定样本上的准确率"""
    all_predictions = []
    batch_size = config.batch_size

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        predictions = predict_batch(batch_texts, model, tokenizer, device, config)
        all_predictions.extend(predictions)

    correct = sum(1 for p, l in zip(all_predictions, labels) if p == l)
    accuracy = correct / len(labels)

    return accuracy

def run_sampling_experiment(model, test_texts, test_labels, tokenizer, device, config,
                            sample_sizes, num_trials=10):
    """
    运行抽样实验

    参数:
        sample_sizes: 不同的样本大小列表
        num_trials: 每个样本大小重复抽样的次数
    """
    results = {
        'sample_size': [],
        'trial': [],
        'accuracy': []
    }

    total_test_size = len(test_texts)

    for sample_size in sample_sizes:
        print(f"\n{'='*60}")
        print(f"Testing sample size: {sample_size}")
        print(f"{'='*60}")

        accuracies = []

        for trial in tqdm(range(num_trials), desc=f"Sample size {sample_size}"):
            # 随机抽样
            if sample_size >= total_test_size:
                sampled_texts = test_texts
                sampled_labels = test_labels
            else:
                indices = random.sample(range(total_test_size), sample_size)
                sampled_texts = [test_texts[i] for i in indices]
                sampled_labels = [test_labels[i] for i in indices]

            # 评估准确率
            accuracy = evaluate_sample(model, sampled_texts, sampled_labels,
                                      tokenizer, device, config)

            results['sample_size'].append(sample_size)
            results['trial'].append(trial + 1)
            results['accuracy'].append(accuracy)
            accuracies.append(accuracy)

        # 计算统计量
        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        min_acc = np.min(accuracies)
        max_acc = np.max(accuracies)

        print(f"\nResults for sample size {sample_size}:")
        print(f"  Mean Accuracy:     {mean_acc:.4f} ({mean_acc*100:.2f}%)")
        print(f"  Std Deviation:     {std_acc:.4f} ({std_acc*100:.2f}%)")
        print(f"  Min Accuracy:      {min_acc:.4f} ({min_acc*100:.2f}%)")
        print(f"  Max Accuracy:      {max_acc:.4f} ({max_acc*100:.2f}%)")
        print(f"  Range:             {(max_acc-min_acc):.4f} ({(max_acc-min_acc)*100:.2f}%)")
        print(f"  95% CI (approx):   ± {1.96*std_acc:.4f} ({1.96*std_acc*100:.2f}%)")

    return pd.DataFrame(results)

def plot_results(df, output_path):
    """绘制结果图表"""
    sample_sizes = sorted(df['sample_size'].unique())

    # 计算统计量
    stats = df.groupby('sample_size')['accuracy'].agg(['mean', 'std', 'min', 'max'])

    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Test Sample Size Impact on Accuracy Estimation Stability',
                 fontsize=16, fontweight='bold')

    # 1. 箱线图
    ax1 = axes[0, 0]
    data_to_plot = [df[df['sample_size']==size]['accuracy'].values for size in sample_sizes]
    ax1.boxplot(data_to_plot, labels=sample_sizes)
    ax1.set_xlabel('Sample Size', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title('Accuracy Distribution by Sample Size', fontsize=13)
    ax1.grid(True, alpha=0.3)

    # 2. 均值和置信区间
    ax2 = axes[0, 1]
    means = stats['mean'].values
    stds = stats['std'].values
    ax2.errorbar(sample_sizes, means, yerr=1.96*stds, marker='o', capsize=5,
                 linewidth=2, markersize=8, label='Mean ± 95% CI')
    ax2.fill_between(sample_sizes, means - 1.96*stds, means + 1.96*stds, alpha=0.2)
    ax2.set_xlabel('Sample Size', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Mean Accuracy with 95% Confidence Interval', fontsize=13)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. 标准差变化
    ax3 = axes[1, 0]
    ax3.plot(sample_sizes, stds, marker='s', linewidth=2, markersize=8, color='red')
    ax3.set_xlabel('Sample Size', fontsize=12)
    ax3.set_ylabel('Standard Deviation', fontsize=12)
    ax3.set_title('Accuracy Standard Deviation vs Sample Size', fontsize=13)
    ax3.grid(True, alpha=0.3)

    # 4. 范围变化
    ax4 = axes[1, 1]
    ranges = (stats['max'] - stats['min']).values
    ax4.plot(sample_sizes, ranges, marker='^', linewidth=2, markersize=8, color='green')
    ax4.set_xlabel('Sample Size', fontsize=12)
    ax4.set_ylabel('Accuracy Range (Max - Min)', fontsize=12)
    ax4.set_title('Accuracy Range vs Sample Size', fontsize=13)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Plot saved to: {output_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Analyze test sample size impact on accuracy stability')
    parser.add_argument('--sample-sizes', nargs='+', type=int,
                        default=[100, 200, 500, 1000],
                        help='List of sample sizes to test')
    parser.add_argument('--trials', type=int, default=10,
                        help='Number of trials per sample size')
    args = parser.parse_args()

    # 设置随机种子以保证可复现性
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # 加载配置
    config = Config()

    # 加载tokenizer
    print("Loading tokenizer...")
    tokenizer = BertTokenizer.from_pretrained(config.model_name)

    # 加载模型
    print("Loading trained model...")
    model = load_trained_model(device, config)

    # 加载测试数据
    print("Loading test data...")
    data_loader = DataLoaderClass(config)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    test_file_path = os.path.join(current_dir, "test.csv")

    try:
        test_texts, test_labels = data_loader.load_csv(test_file_path)
    except FileNotFoundError:
        print(f"Error: {test_file_path} not found.")
        return

    print(f"Total test samples: {len(test_texts)}")

    # 运行实验
    print("\n" + "="*60)
    print("Starting Sampling Stability Experiment")
    print("="*60)

    df_results = run_sampling_experiment(
        model, test_texts, test_labels, tokenizer, device, config,
        sample_sizes=args.sample_sizes,
        num_trials=args.trials
    )

    # 保存结果
    output_dir = current_dir
    csv_path = os.path.join(output_dir, "sample_stability_results.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"\n✓ Results saved to: {csv_path}")

    # 绘制图表
    plot_path = os.path.join(output_dir, "sample_stability_plot.png")
    plot_results(df_results, plot_path)

    # 打印总结
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    summary = df_results.groupby('sample_size')['accuracy'].agg(['mean', 'std', 'min', 'max'])
    summary['range'] = summary['max'] - summary['min']
    summary['cv'] = summary['std'] / summary['mean']  # 变异系数
    print(summary)

    print("\n✓ Experiment completed!")

if __name__ == "__main__":
    main()