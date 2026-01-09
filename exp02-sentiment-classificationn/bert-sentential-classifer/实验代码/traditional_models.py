"""
传统机器学习模型 - 情感分类
实现SVM、朴素贝叶斯、逻辑回归等传统机器学习模型
用于与BERT模型进行对比
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import argparse
import pickle
import os
from tqdm import tqdm
import time

from load_data import DataLoader as DataLoaderClass
from config import Config

class TraditionalModels:
    """传统机器学习模型集合"""

    def __init__(self, max_features=5000):
        """
        初始化

        参数:
            max_features: TF-IDF特征的最大数量
        """
        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),  # 使用unigram和bigram
            min_df=2,             # 忽略出现次数过少的词
            max_df=0.8            # 忽略出现次数过多的词
        )

        # 定义模型
        self.models = {
            'Logistic Regression': LogisticRegression(
                max_iter=1000,
                random_state=42,
                solver='lbfgs'
            ),
            'SVM (Linear)': SVC(
                kernel='linear',
                probability=True,
                random_state=42
            ),
            'Naive Bayes': MultinomialNB(
                alpha=1.0
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            )
        }

        self.trained_models = {}
        self.results = {}

    def fit_vectorizer(self, texts):
        """训练TF-IDF向量化器"""
        print("训练TF-IDF向量化器...")
        self.vectorizer.fit(texts)
        print(f"✓ 提取了 {len(self.vectorizer.vocabulary_)} 个特征")

    def transform_texts(self, texts):
        """将文本转换为TF-IDF特征"""
        return self.vectorizer.transform(texts)

    def train_model(self, model_name, X_train, y_train):
        """训练单个模型"""
        print(f"\n训练 {model_name}...")
        model = self.models[model_name]

        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time

        self.trained_models[model_name] = model
        print(f"✓ 训练完成 (耗时: {train_time:.2f}秒)")

        return train_time

    def evaluate_model(self, model_name, X_test, y_test):
        """评估单个模型"""
        model = self.trained_models[model_name]

        # 预测
        start_time = time.time()
        y_pred = model.predict(X_test)
        pred_time = time.time() - start_time

        # 计算指标
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='binary')
        recall = recall_score(y_test, y_pred, average='binary')
        f1 = f1_score(y_test, y_pred, average='binary')

        # 计算概率（用于AUC）
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, 'decision_function'):
            y_prob = model.decision_function(X_test)
        else:
            y_prob = y_pred

        results = {
            'model': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'train_time': self.results.get(model_name, {}).get('train_time', 0),
            'pred_time': pred_time,
            'y_pred': y_pred,
            'y_prob': y_prob
        }

        self.results[model_name] = results
        return results

    def save_models(self, save_dir='traditional_models'):
        """保存训练好的模型"""
        os.makedirs(save_dir, exist_ok=True)

        # 保存向量化器
        vectorizer_path = os.path.join(save_dir, 'vectorizer.pkl')
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(self.vectorizer, f)
        print(f"✓ 向量化器已保存: {vectorizer_path}")

        # 保存每个模型
        for model_name, model in self.trained_models.items():
            safe_name = model_name.replace(' ', '_').replace('(', '').replace(')', '').lower()
            model_path = os.path.join(save_dir, f'{safe_name}.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            print(f"✓ {model_name} 已保存: {model_path}")

    def load_models(self, save_dir='traditional_models'):
        """加载训练好的模型"""
        # 加载向量化器
        vectorizer_path = os.path.join(save_dir, 'vectorizer.pkl')
        with open(vectorizer_path, 'rb') as f:
            self.vectorizer = pickle.load(f)
        print(f"✓ 向量化器已加载: {vectorizer_path}")

        # 加载每个模型
        for model_name in self.models.keys():
            safe_name = model_name.replace(' ', '_').replace('(', '').replace(')', '').lower()
            model_path = os.path.join(save_dir, f'{safe_name}.pkl')
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    self.trained_models[model_name] = pickle.load(f)
                print(f"✓ {model_name} 已加载: {model_path}")

def generate_comparison_report(results, output_path='traditional_models_report.txt'):
    """生成对比报告"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("传统机器学习模型 - 情感分类评估报告\n")
        f.write("="*70 + "\n\n")

        f.write("【模型性能对比】\n")
        f.write("-"*70 + "\n")
        f.write(f"{'模型':<25} {'准确率':>10} {'精确率':>10} {'召回率':>10} {'F1分数':>10}\n")
        f.write("-"*70 + "\n")

        for model_name, result in results.items():
            f.write(f"{model_name:<25} "
                   f"{result['accuracy']:>10.4f} "
                   f"{result['precision']:>10.4f} "
                   f"{result['recall']:>10.4f} "
                   f"{result['f1_score']:>10.4f}\n")

        f.write("\n【训练时间对比】\n")
        f.write("-"*70 + "\n")
        f.write(f"{'模型':<25} {'训练时间':>15} {'预测时间':>15}\n")
        f.write("-"*70 + "\n")

        for model_name, result in results.items():
            f.write(f"{model_name:<25} "
                   f"{result['train_time']:>14.2f}秒 "
                   f"{result['pred_time']:>14.4f}秒\n")

        # 找出最佳模型
        f.write("\n【最佳模型】\n")
        f.write("-"*70 + "\n")

        best_acc_model = max(results.items(), key=lambda x: x[1]['accuracy'])
        best_f1_model = max(results.items(), key=lambda x: x[1]['f1_score'])

        f.write(f"最高准确率: {best_acc_model[0]} ({best_acc_model[1]['accuracy']:.4f})\n")
        f.write(f"最高F1分数: {best_f1_model[0]} ({best_f1_model[1]['f1_score']:.4f})\n")

        f.write("\n【模型特点】\n")
        f.write("-"*70 + "\n")
        f.write("• 逻辑回归 (Logistic Regression): 训练快速，可解释性强\n")
        f.write("• SVM (线性核): 适合高维数据，泛化能力强\n")
        f.write("• 朴素贝叶斯 (Naive Bayes): 训练极快，对小样本友好\n")
        f.write("• 随机森林 (Random Forest): 鲁棒性强，不易过拟合\n")

        f.write("="*70 + "\n")

    print(f"\n✓ 报告已保存: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='训练传统机器学习模型')
    parser.add_argument('--max-features', type=int, default=5000,
                       help='TF-IDF特征最大数量')
    parser.add_argument('--max-train-samples', type=int, default=10000,
                       help='训练样本数量限制')
    parser.add_argument('--save-models', action='store_true',
                       help='保存训练好的模型')
    parser.add_argument('--output-dir', type=str, default='.',
                       help='输出目录')
    args = parser.parse_args()

    print("="*70)
    print("传统机器学习模型训练与评估")
    print("="*70 + "\n")

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

    # 限制训练样本数量
    if len(train_texts) > args.max_train_samples:
        print(f"限制训练样本: {len(train_texts)} -> {args.max_train_samples}")
        train_texts = train_texts[:args.max_train_samples]
        train_labels = train_labels[:args.max_train_samples]

    print(f"训练集: {len(train_texts)} 样本")
    print(f"测试集: {len(test_texts)} 样本\n")

    # 初始化模型
    tm = TraditionalModels(max_features=args.max_features)

    # 训练TF-IDF向量化器
    tm.fit_vectorizer(train_texts)

    # 转换文本为特征
    print("\n转换文本为TF-IDF特征...")
    X_train = tm.transform_texts(train_texts)
    X_test = tm.transform_texts(test_texts)
    y_train = np.array(train_labels)
    y_test = np.array(test_labels)

    print(f"训练集特征维度: {X_train.shape}")
    print(f"测试集特征维度: {X_test.shape}")

    # 训练所有模型
    print("\n" + "="*70)
    print("开始训练所有模型...")
    print("="*70)

    for model_name in tm.models.keys():
        train_time = tm.train_model(model_name, X_train, y_train)
        tm.results[model_name] = {'train_time': train_time}

    # 评估所有模型
    print("\n" + "="*70)
    print("评估所有模型...")
    print("="*70)

    results = {}
    for model_name in tm.trained_models.keys():
        print(f"\n评估 {model_name}...")
        result = tm.evaluate_model(model_name, X_test, y_test)
        results[model_name] = result

        print(f"  准确率:   {result['accuracy']:.4f}")
        print(f"  精确率:   {result['precision']:.4f}")
        print(f"  召回率:   {result['recall']:.4f}")
        print(f"  F1分数:   {result['f1_score']:.4f}")

    # 保存模型（如果指定）
    if args.save_models:
        print("\n保存模型...")
        save_dir = os.path.join(args.output_dir, 'traditional_models')
        tm.save_models(save_dir)

    # 生成对比报告
    report_path = os.path.join(args.output_dir, 'traditional_models_report.txt')
    generate_comparison_report(results, report_path)

    # 保存结果到CSV
    results_df = pd.DataFrame([
        {
            '模型': name,
            '准确率': res['accuracy'],
            '精确率': res['precision'],
            '召回率': res['recall'],
            'F1分数': res['f1_score'],
            '训练时间(秒)': res['train_time'],
            '预测时间(秒)': res['pred_time']
        }
        for name, res in results.items()
    ])

    csv_path = os.path.join(args.output_dir, 'traditional_models_results.csv')
    results_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"✓ 结果已保存: {csv_path}")

    print("\n✅ 所有模型训练和评估完成！")
    print("\n最佳模型:")
    best_model = max(results.items(), key=lambda x: x[1]['f1_score'])
    print(f"  {best_model[0]}: F1={best_model[1]['f1_score']:.4f}")

if __name__ == "__main__":
    main()
