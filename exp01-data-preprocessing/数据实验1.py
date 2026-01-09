import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk
import re

# 下载必要的nltk数据
nltk.download('punkt')


def preprocess_text(text):
    """文本预处理函数，先兜底保证是字符串"""
    import math

    # 处理 None / NaN / float 等情况，统一转成干净的字符串
    if text is None:
        text = ""
    elif isinstance(text, float):
        if math.isnan(text):
            text = ""
        else:
            text = str(text)
    else:
        text = str(text)

    # 转换为小写
    text = text.lower()
    # 移除特殊字符
    text = re.sub(r'[^\w\s]', '', text)
    # 分词
    tokens = word_tokenize(text)
    return tokens


def load_and_preprocess_data(file_path):
    """加载并预处理数据"""
    # 读取CSV文件
    df = pd.read_csv(file_path)

    # 合并标题和评论
    df['text'] = df.iloc[:, 1].astype(str).fillna('') + " " + df.iloc[:, 2].astype(str).fillna('')

    # 预处理所有文本
    corpus = []
    for text in df['text']:
        tokens = preprocess_text(text)
        corpus.append(tokens)

    return corpus, df.iloc[:, 0].values, df  # 返回处理后的文本、标签和带 text 列的 df


def train_word2vec(corpus):
    """训练Word2Vec模型"""
    model = Word2Vec(sentences=corpus,
                     vector_size=100,  # 词向量维度
                     window=5,  # 上下文窗口大小
                     min_count=1,  # 词频阈值
                     workers=4)  # 训练的线程数
    return model


def get_document_vector(text, model):
    """获取文档的词向量表示（取平均）"""
    tokens = preprocess_text(text)
    vectors = []
    for token in tokens:
        if token in model.wv:
            vectors.append(model.wv[token])
    if vectors:
        return np.mean(vectors, axis=0)
    return np.zeros(model.vector_size)


def main():
    # 加载数据（同时拿到带 text 列的 df）
    corpus, labels, df = load_and_preprocess_data('test.csv')

    # 训练Word2Vec模型
    model = train_word2vec(corpus)

    # 获取文档向量
    doc_vectors = []
    for text in df['text']:
        doc_vector = get_document_vector(text, model)
        doc_vectors.append(doc_vector)

    # 转换为numpy数组
    X = np.array(doc_vectors)
    y = labels

    print("文档向量形状:", X.shape)
    print("标签形状:", y.shape)

    # 保存模型（可选）
    model.save("word2vec_sentiment.model")

    # 示例：查看某些词的相似词
    word = "great"
    if word in model.wv:
        similar_words = model.wv.most_similar(word)
        print(f"\n与'{word}'最相似的词:")
        for word, score in similar_words:
            print(f"{word}: {score:.4f}")


if __name__ == "__main__":
    main()