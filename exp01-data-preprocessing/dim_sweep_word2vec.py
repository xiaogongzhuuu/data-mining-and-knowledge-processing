import os
import math
import re
import time

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
import matplotlib

# 确保分词资源就绪
nltk.download("punkt")

# -------- 0. （可选）中文字体，方便画图中文标题 --------
def set_chinese_font():
    import matplotlib.font_manager
    font_list = ['PingFang SC', 'Heiti SC', 'STHeiti', 'SimHei', 'Microsoft YaHei', 'SimSun']
    available_fonts = set(f.name for f in matplotlib.font_manager.fontManager.ttflist)
    for font in font_list:
        if font in available_fonts:
            matplotlib.rcParams['font.sans-serif'] = [font]
            matplotlib.rcParams['axes.unicode_minus'] = False
            print(f"✅ 已启用中文字体: {font}")
            return
    print("⚠️ 未检测到常用中文字体，图表中文可能会显示乱码。")


# -------- 1. 文本预处理（和 classification_wordvec2.py 基本一致） --------
def preprocess_text(text):
    if text is None:
        text = ""
    elif isinstance(text, float):
        text = "" if math.isnan(text) else str(text)
    else:
        text = str(text)
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    tokens = word_tokenize(text)
    return tokens


# -------- 2. 读数据 --------
def load_data(csv_name="test.csv", max_samples=None):
    """
    max_samples: 为了加快实验，可以只取前 max_samples 条来做维度对比
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, csv_name)
    df = pd.read_csv(csv_path)

    # 合并 title + review
    df["text"] = (
        df.iloc[:, 1].astype(str).fillna("") + " " +
        df.iloc[:, 2].astype(str).fillna("")
    )

    if max_samples is not None and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)

    labels_raw = df.iloc[:, 0].values
    y = np.array([0 if lab == 1 else 1 for lab in labels_raw], dtype=int)
    texts = df["text"].tolist()
    return texts, y


# -------- 3. 文档向量：平均词向量 --------
def get_document_vector(tokens, model, vector_size):
    vectors = []
    for w in tokens:
        if w in model.wv:
            vectors.append(model.wv[w])
    if not vectors:
        return np.zeros(vector_size, dtype=float)
    return np.mean(vectors, axis=0)


def main():
    set_chinese_font()
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # 1. 加载数据（可以先用较少样本快速对比，比如 100000，如果你想全量就改成 None）
    print("👉 加载数据，用于维度对比实验...")
    texts, y = load_data("test.csv", max_samples=100000)
    print("  样本数:", len(texts))

    # 2. 先把文本预处理 + 分好词，方便重复利用
    print("👉 文本预处理 & 分词...")
    corpus = [preprocess_text(t) for t in texts]

    # 3. 预先划分好 train/test 索引，保证不同维度实验可比
    print("👉 划分统一的训练/测试索引...")
    indices = np.arange(len(y))
    idx_train, idx_test, y_train, y_test = train_test_split(
        indices, y, test_size=0.2, random_state=42, stratify=y
    )

    dims = [50, 100, 200, 300]
    results = []

    for dim in dims:
        print("\n" + "=" * 60)
        print(f"🔢 正在实验向量维度 dim = {dim}")
        start_time = time.time()

        # 3.1 训练 Word2Vec
        print("👉 训练 Word2Vec 模型...")
        model = Word2Vec(
            sentences=corpus,
            vector_size=dim,
            window=5,
            min_count=1,
            workers=4,    # 这里可以多线程加速
            sg=1,         # 1=skip-gram, 0=CBOW，你可以按需要改
        )

        # 3.2 构造文档向量矩阵 X
        print("👉 构造文档向量...")
        doc_vectors = [
            get_document_vector(tokens, model, vector_size=dim)
            for tokens in corpus
        ]
        X = np.array(doc_vectors)

        # 3.3 按统一索引划分 train/test
        X_train = X[idx_train]
        X_test = X[idx_test]

        # 3.4 标准化
        scaler = StandardScaler()
        X_train_std = scaler.fit_transform(X_train)
        X_test_std = scaler.transform(X_test)

        # 3.5 训练 Logistic 回归
        print("👉 训练 Logistic Regression...")
        clf = LogisticRegression(
            max_iter=1000,
            solver="liblinear",
            random_state=42,
        )
        clf.fit(X_train_std, y_train)

        # 3.6 评估
        y_pred = clf.predict(X_test_std)
        acc = accuracy_score(y_test, y_pred)
        elapsed = time.time() - start_time
        print(f"🎯 dim = {dim} 的测试集 Accuracy: {acc:.4f}，耗时约 {elapsed:.1f} 秒")

        results.append((dim, acc, elapsed))

    # 4. 打印结果表
    print("\n" + "=" * 60)
    print("维度 vs 准确率 对比结果：")
    for dim, acc, elapsed in results:
        print(f"  dim = {dim:3d}  ->  Accuracy = {acc:.4f}，耗时约 {elapsed:.1f} 秒")

    # 5. 画维度-准确率折线图
    dims_list = [r[0] for r in results]
    acc_list = [r[1] for r in results]

    plt.figure(figsize=(7, 5), dpi=120)
    plt.plot(dims_list, acc_list, marker="o")
    plt.title("词向量维度 vs 分类准确率", fontsize=14)
    plt.xlabel("词向量维度 (vector_size)", fontsize=12)
    plt.ylabel("测试集 Accuracy", fontsize=12)
    plt.grid(True)
    plt.tight_layout()

    out_path = os.path.join(base_dir, "dim_vs_accuracy.png")
    plt.savefig(out_path, dpi=300)
    print(f"✅ 维度-准确率折线图已保存到: {out_path}")
    plt.show()


if __name__ == "__main__":
    main()