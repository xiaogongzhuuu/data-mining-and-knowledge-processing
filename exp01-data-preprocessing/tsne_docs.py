import os
import math
import re
import multiprocessing
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# 确保 tokenizer 可用
nltk.download("punkt")

# 全局变量，用于子进程共享模型
global_model_wv = None

def init_worker(model_path):
    """子进程初始化：加载模型"""
    global global_model_wv
    print(f"🔧 子进程 {os.getpid()} 正在加载模型...")
    # 只加载 KeyedVectors 以节省内存（如果只需要向量）
    # 注意：如果 model 保存的是完整 Word2Vec 对象，load 后取 .wv
    model = Word2Vec.load(model_path)
    global_model_wv = model.wv
    # 锁定以防意外修改
    global_model_wv.init_sims(replace=True)

def preprocess_text(text):
    """和你训练 Word2Vec 时同源的预处理逻辑，保证一致"""
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
    return tokens

def get_document_vector_worker(text):
    """Worker 函数：计算单个文本的向量"""
    global global_model_wv
    tokens = preprocess_text(text)
    vecs = []
    for tok in tokens:
        if tok in global_model_wv:
            vecs.append(global_model_wv[tok])
    if vecs:
        return np.mean(vecs, axis=0)
    # 没有任何词在词表里，就给一个全零向量
    return np.zeros(global_model_wv.vector_size, dtype=np.float32)

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    w2v_path = os.path.join(base_dir, "word2vec_sentiment.model")
    
    # === 1. 加载评论数据 ===
    csv_path = os.path.join(base_dir, "test.csv")
    print("👉 正在加载评论数据:", csv_path)
    df = pd.read_csv(csv_path, header=None, names=["label", "title", "text"])
    
    # 合并 title + text 作为完整评论
    df["full_text"] = df["title"].astype(str).fillna("") + " " + df["text"].astype(str).fillna("")
    print("👉 评论总数:", len(df))

    # === 2. 计算或加载文档向量 (Caching) ===
    cache_X_path = os.path.join(base_dir, "cache_X_all.npy")
    cache_y_path = os.path.join(base_dir, "cache_y_all.npy")

    if os.path.exists(cache_X_path) and os.path.exists(cache_y_path):
        print("⚡️ 发现缓存文件，正在加载...")
        X_all = np.load(cache_X_path)
        y_all = np.load(cache_y_path)
        print("✅ 加载完成")
    else:
        print("🚀 未发现缓存，开始并行计算文档向量...")
        
        # 准备数据
        texts = df["full_text"].tolist()
        labels = df["label"].values
        
        # 并行计算
        # 根据 CPU 核心数决定进程数，保留一个核心给系统
        num_workers = max(1, multiprocessing.cpu_count() - 1)
        print(f"👉 启动 {num_workers} 个进程进行计算...")
        
        with multiprocessing.Pool(processes=num_workers, initializer=init_worker, initargs=(w2v_path,)) as pool:
            # 使用 imap 稍微节省内存，并显示进度
            doc_vectors = []
            total = len(texts)
            for i, vec in enumerate(pool.imap(get_document_vector_worker, texts, chunksize=100)):
                doc_vectors.append(vec)
                if (i + 1) % 10000 == 0:
                    print(f"   已处理 {i + 1}/{total} 条评论...")
        
        X_all = np.vstack(doc_vectors)
        y_all = labels
        
        print("💾 正在保存缓存...")
        np.save(cache_X_path, X_all)
        np.save(cache_y_path, y_all)
        print("✅ 缓存已保存")

    print("✅ 文档向量矩阵形状:", X_all.shape)

    # === 3. 是否抽样 ===
    N_SAMPLES = 200000  # 你的目标是 20万
    if N_SAMPLES is not None and X_all.shape[0] > N_SAMPLES:
        print(f"👉 评论太多，只随机采样 {N_SAMPLES} 条用于 t-SNE")
        idx = np.random.choice(X_all.shape[0], N_SAMPLES, replace=False)
        X = X_all[idx]
        y = y_all[idx]
    else:
        print("👉 使用全部评论做 t-SNE")
        X = X_all
        y = y_all

    print("👉 参与 t-SNE 的评论数:", X.shape[0])

    # === 4. t-SNE 降维 (优化参数) ===
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        init="pca",          # ⚡️ 优化：使用 PCA 初始化，通常更快且效果更好
        learning_rate="auto", # ⚡️ 优化：自动学习率
        n_jobs=-1,           # ⚡️ 优化：使用所有核心进行最近邻搜索
        random_state=42,
        verbose=1            # 显示进度
    )
    print("👉 开始对评论向量做 t-SNE 降维 ...")
    X_2d = tsne.fit_transform(X)
    print("✅ t-SNE 完成")

    # === 5. 画散点图 ===
    plt.figure(figsize=(10, 8)) # 稍微大一点
    
    neg_mask = (y == 1)
    pos_mask = (y == 2)

    # 降低 alpha 和点大小以应对大量数据
    plt.scatter(X_2d[neg_mask, 0], X_2d[neg_mask, 1], c="blue", s=1, alpha=0.3, label="Negative")
    plt.scatter(X_2d[pos_mask, 0], X_2d[pos_mask, 1], c="red",  s=1, alpha=0.3, label="Positive")

    plt.legend(markerscale=5) # 图例的点放大一点方便看
    plt.title(f"t-SNE of Amazon Review Vectors (n={X.shape[0]})")
    plt.tight_layout()

    out_path = os.path.join(base_dir, "tsne_docs_all_optimized.png")
    plt.savefig(out_path, dpi=300)
    print("✅ 评论级 t-SNE 图已保存到:", out_path)

if __name__ == "__main__":
    # Mac 上 multiprocessing 需要这个
    multiprocessing.freeze_support()
    main()