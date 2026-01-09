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
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

# 确保分词资源就绪
nltk.download("punkt")


# -------- 0. Mac/Windows 中文字体自动适配设置 --------
def set_chinese_font():
    """
    自动选择系统支持的中文字体，优先适配 Mac
    """
    # 常见中文字体列表，优先级：Mac苹方 -> Mac黑体 -> Win黑体 -> Win雅黑
    font_list = ['PingFang SC', 'Heiti SC', 'STHeiti', 'SimHei', 'Microsoft YaHei', 'SimSun']

    # 获取当前系统所有可用字体
    import matplotlib.font_manager
    available_fonts = set(f.name for f in matplotlib.font_manager.fontManager.ttflist)

    found_font = False
    for font in font_list:
        if font in available_fonts:
            matplotlib.rcParams['font.sans-serif'] = [font]
            matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题
            print(f"✅ 已启用中文字体: {font}")
            found_font = True
            break

    if not found_font:
        # 如果都没找到，尝试设置通用 sans-serif，并警告
        matplotlib.rcParams['font.sans-serif'] = ['sans-serif']
        print("⚠️ 未检测到常用中文字体，图表中文可能会显示乱码，请检查系统字体库。")


# -------- 1. 文本预处理 --------
def preprocess_text(text):
    if text is None:
        text = ""
    elif isinstance(text, float):
        text = "" if math.isnan(text) else str(text)
    else:
        text = str(text)
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return word_tokenize(text)


# -------- 2. 读数据 --------
def load_data(csv_name="test.csv"):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, csv_name)
    df = pd.read_csv(csv_path)
    df["text"] = (df.iloc[:, 1].astype(str).fillna("") + " " + df.iloc[:, 2].astype(str).fillna(""))
    labels_raw = df.iloc[:, 0].values
    y = np.array([0 if lab == 1 else 1 for lab in labels_raw], dtype=int)
    return df["text"].tolist(), y


# -------- 3. 文档向量化 --------
def get_document_vector(tokens, model, vector_size=100):
    vectors = []
    for w in tokens:
        if w in model.wv:
            vectors.append(model.wv[w])
    if not vectors:
        return np.zeros(vector_size, dtype=float)
    return np.mean(vectors, axis=0)


# -------- 4. 主流程 --------
def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # 4.1 加载模型
    model_path = os.path.join(base_dir, "word2vec_sentiment.model")
    print(f"👉 正在加载 Word2Vec 模型: {model_path}")
    try:
        model = Word2Vec.load(model_path)
    except FileNotFoundError:
        print("❌ 找不到模型文件，请先运行实验一训练代码。")
        return
    vector_size = model.vector_size

    # 4.2 加载数据
    print("👉 正在加载文本和标签...")
    texts, y = load_data("test.csv")

    # 4.3 构造向量
    print("👉 正在构造文档向量 X ...")
    doc_vectors = [get_document_vector(preprocess_text(t), model, vector_size) for t in texts]
    X = np.array(doc_vectors)

    # 4.4 划分数据集
    print("👉 划分训练/测试集...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4.5 标准化
    print("👉 特征标准化...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 4.6 训练
    print("👉 训练 Logistic Regression...")
    clf = LogisticRegression(max_iter=1000, n_jobs=-1, solver="liblinear", random_state=42)
    clf.fit(X_train, y_train)

    # 4.7 评估
    print("👉 评估模型...")
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n🎯 测试集 Accuracy: {acc:.4f}\n")
    print(classification_report(y_test, y_pred, target_names=["Negative", "Positive"], digits=4))

    # =======================================================
    # 👇👇👇 重点优化：绘图部分 👇👇👇
    # =======================================================

    # 设置中文字体
    set_chinese_font()

    cm = confusion_matrix(y_test, y_pred)

    # 1. 构造高级标签 (Label + Count + Percentage)
    # 对应混淆矩阵的四个格子：[TN, FP], [FN, TP]
    group_names = [
        '真负类 (TN)\n正确预测差评',  # 0,0
        '假正类 (FP)\n差评误判为好评',  # 0,1
        '假负类 (FN)\n好评误判为差评',  # 1,0
        '真正类 (TP)\n正确预测好评'  # 1,1
    ]

    group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in cm.flatten() / np.sum(cm)]

    # 组合文字
    labels = [f"{v1}\n{v2}\n({v3})" for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2, 2)

    # 2. 绘图
    plt.figure(figsize=(9, 7), dpi=120)  # 增加尺寸和分辨率

    # 使用 seaborn heatmap
    ax = sns.heatmap(
        cm,
        annot=labels,
        fmt='',  # 必须为空，因为我们手动构造了 labels 字符串
        cmap='Blues',  # 蓝色系，专业且清晰
        cbar=True,
        xticklabels=["预测为差评 (Neg)", "预测为好评 (Pos)"],
        yticklabels=["实际为差评 (Neg)", "实际为好评 (Pos)"],
        annot_kws={"size": 11, "weight": "bold"}  # 字体加粗
    )

    # 3. 调整标题和轴标签
    plt.title(f"情感分类混淆矩阵\n(Logistic Regression, Accuracy: {acc:.2%})", fontsize=15, fontweight='bold', pad=20)
    plt.xlabel("模型预测结果", fontsize=12, labelpad=10)
    plt.ylabel("真实标签", fontsize=12, labelpad=10)

    # 调整刻度字体
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10, rotation=0)  # y轴文字横向显示

    plt.tight_layout()

    out_path = os.path.join(base_dir, "confusion_matrix_final.png")
    plt.savefig(out_path, dpi=300)
    print(f"✅ 优化后的混淆矩阵已保存到: {out_path}")
    plt.show()


if __name__ == "__main__":
    main()