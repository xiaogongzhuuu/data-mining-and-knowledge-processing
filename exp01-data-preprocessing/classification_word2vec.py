import os
import math
import re

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
)
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# 确保分词资源就绪
nltk.download("punkt")

# -------- 1. 文本预处理（建议和你前面保持一致，可以按需再加 stopwords） --------

def preprocess_text(text):
    """简单版本：小写、去标点、分词"""
    if text is None:
        text = ""
    elif isinstance(text, float):
        if math.isnan(text):
            text = ""
        else:
            text = str(text)
    else:
        text = str(text)

    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    tokens = word_tokenize(text)

    # 这里可以按需要复制你在 wordcloud_sentiment.py 里那套更严格的过滤逻辑
    # 比如只保留长度>2、字母词、去停用词等
    return tokens


# -------- 2. 读数据 + 合并 title & review --------

def load_data(csv_name="test.csv"):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, csv_name)

    df = pd.read_csv(csv_path)

    # 第0列：label (1=neg, 2=pos)，第1列：title，第2列：review
    df["text"] = (
        df.iloc[:, 1].astype(str).fillna("") + " " +
        df.iloc[:, 2].astype(str).fillna("")
    )

    # 把 label 变成 0 / 1 方便 sklearn 使用
    labels_raw = df.iloc[:, 0].values
    # Amazon polarity: 1 -> negative(0), 2 -> positive(1)
    y = np.array([0 if lab == 1 else 1 for lab in labels_raw], dtype=int)

    return df["text"].tolist(), y


# -------- 3. 把一条评论变成“平均词向量” --------

def get_document_vector(tokens, model, vector_size=100):
    """对一条评论的词向量取平均，如果全是 OOV 词则返回零向量"""
    vectors = []
    for w in tokens:
        if w in model.wv:
            vectors.append(model.wv[w])
    if not vectors:
        return np.zeros(vector_size, dtype=float)
    return np.mean(vectors, axis=0)


# -------- 4. 主流程：加载模型 -> 构造 X, y -> 训练分类器 -> 画混淆矩阵 --------

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # 4.1 加载你之前训练好的 Word2Vec 模型
    model_path = os.path.join(base_dir, "word2vec_sentiment.model")
    print(f"👉 正在加载 Word2Vec 模型: {model_path}")
    model = Word2Vec.load(model_path)
    vector_size = model.vector_size
    print("✅ 模型加载完成，向量维度:", vector_size)

    # 4.2 加载文本和标签
    print("👉 正在加载文本和标签...")
    texts, y = load_data("test.csv")
    print("  样本数:", len(texts))

    # 4.3 文本预处理 + 文档向量
    print("👉 正在构造文档向量 X ...（可能稍微有点慢）")
    doc_vectors = []
    for i, text in enumerate(texts):
        tokens = preprocess_text(text)
        vec = get_document_vector(tokens, model, vector_size=vector_size)
        doc_vectors.append(vec)
        # 可选：看看进度
        # if (i + 1) % 5000 == 0:
        #     print(f"  已处理 {i+1} 条")

    X = np.array(doc_vectors)
    print("✅ 文档向量形状:", X.shape)
    print("✅ 标签形状:", y.shape)

    # 4.4 划分训练集 / 测试集
    print("👉 划分训练/测试集...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print("  训练集大小:", X_train.shape[0])
    print("  测试集大小:", X_test.shape[0])

    # 4.5 特征标准化以提高数值稳定性
    print("👉 对特征进行标准化(StandardScaler)...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 4.6 训练 Logistic Regression（指定更稳定的 solver 并固定随机种子）
    print("👉 训练 Logistic Regression 分类器...")
    clf = LogisticRegression(
        max_iter=1000,
        n_jobs=-1,
        solver="liblinear",
        random_state=42,
    )
    clf.fit(X_train, y_train)
    print("✅ 训练完成")

    # 4.6 在测试集上评估
    print("👉 在测试集上评估...")
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n🎯 测试集 Accuracy: {acc:.4f}\n")

    print("👉 分类报告 (precision / recall / f1)：")
    print(classification_report(
        y_test,
        y_pred,
        target_names=["negative", "positive"],
        digits=4,
    ))

    # 4.7 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    print("👉 混淆矩阵原始数值：")
    print(cm)

    # 4.8 绘制混淆矩阵热力图
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Pred Neg", "Pred Pos"],
        yticklabels=["True Neg", "True Pos"],
    )
    plt.title("Confusion Matrix (Logistic Regression on Word2Vec doc vectors)")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()

    out_path = os.path.join(base_dir, "confusion_matrix_word2vec.png")
    plt.savefig(out_path, dpi=300)
    print(f"✅ 混淆矩阵热力图已保存到: {out_path}")
    plt.show()


if __name__ == "__main__":
    main()