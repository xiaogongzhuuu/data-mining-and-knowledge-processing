import os
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import random
import numpy as np

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "word2vec_sentiment.model")

    print("👉 正在加载模型:", model_path)
    model = Word2Vec.load(model_path)
    print("✅ 模型加载完成")

    # 1. 准备要可视化的词
    # 你可以自己调整这几个列表
    positive_words = ["good", "great", "excellent", "amazing", "fantastic", "wonderful", "nice", "awesome", "love"]
    negative_words = ["bad", "terrible", "awful", "horrible", "waste", "poor", "disappointing", "hate", "worst"]
    product_words  = ["book", "camera", "phone", "battery", "screen", "case", "headphones", "laptop"]

    groups = [
        ("Positive", positive_words, "red"),
        ("Negative", negative_words, "blue"),
        ("Product",  product_words,  "green"),
    ]

    words = []
    vectors = []
    colors = []

    for label, word_list, color in groups:
        for w in word_list:
            if w in model.wv:
                words.append(w)
                vectors.append(model.wv[w])
                colors.append(color)
            else:
                print(f"⚠️ 词 '{w}' 不在词表中，跳过")

    if not vectors:
        print("❌ 没有任何词在模型词表中，检查一下训练数据或词表")
        return

    print("👉 参与可视化的词数:", len(words))

    # 2. t-SNE 降维到 2D
    tsne = TSNE(
        n_components=2,
        perplexity=6,      # 样本数不多时用小一点
        init="random",
        learning_rate=200,
        random_state=42
    )
    print("👉 开始 t-SNE 降维 ...")
    X = np.array(vectors)  # ⭐ 关键：转成 (n_samples, dim) 的 numpy 数组
    embedding_2d = tsne.fit_transform(X)
    print("✅ t-SNE 完成")

    # 3. 画图
    plt.figure(figsize=(8, 6))
    for i, (x, y) in enumerate(embedding_2d):
        plt.scatter(x, y, c=colors[i], s=30)
        plt.text(x + 0.01, y + 0.01, words[i], fontsize=9)

    plt.title("t-SNE 可视化：词向量语义空间（Amazon 评论）", fontproperties="SimHei")
    plt.tight_layout()

    # 保存图片
    out_path = os.path.join(base_dir, "tsne_words.png")
    plt.savefig(out_path, dpi=300)
    print("✅ 图像已保存到:", out_path)

    # 如果你想弹窗显示，也可以加上：
    # plt.show()

if __name__ == "__main__":
    main()