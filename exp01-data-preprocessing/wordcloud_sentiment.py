import os
import math
import re
import pandas as pd
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from collections import Counter

# 下载停用词资源（如果已经存在会自动跳过）
nltk.download('punkt')
nltk.download('stopwords')

# 英文通用停用词 + 领域相关无信息词（产品类别、数量词等）
EN_STOPWORDS = set(stopwords.words('english'))
CUSTOM_STOPWORDS = {
    'book', 'books', 'movie', 'film', 'cd', 'dvd', 'album', 'product', 'item',
    'one', 'time', 'story', 'review', 'people', 'game', 'version', 'copy',
    'series', 'music','money','easy','reviews'
}
ALL_STOPWORDS = EN_STOPWORDS | CUSTOM_STOPWORDS

# -------- 1. 文本预处理：和数据实验1.py 保持风格一致 --------

def preprocess_text(text):
    """文本预处理函数，先兜底保证是字符串"""
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
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)

    # 进一步清洗：只保留字母词，长度>2，且不在停用词表中
    clean_tokens = []
    for tok in tokens:
        if not tok.isalpha():
            continue
        if len(tok) <= 2:
            continue
        if tok in ALL_STOPWORDS:
            continue
        clean_tokens.append(tok)

    return clean_tokens


# -------- 2. 读取 CSV，构造 text 列，并按标签划分 --------

def load_data(csv_name="test.csv"):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, csv_name)

    df = pd.read_csv(csv_path)

    # 第 0 列：标签（Amazon polarity: 1=negative, 2=positive）
    # 第 1 列：标题，第 2 列：评论正文
    df["text"] = (
        df.iloc[:, 1].astype(str).fillna("") + " " +
        df.iloc[:, 2].astype(str).fillna("")
    )

    # 按标签拆分：你可以根据自己的理解改这两行
    neg_df = df[df.iloc[:, 0] == 1]
    pos_df = df[df.iloc[:, 0] == 2]

    # 为了画图速度，可以采样一部分
    def sample_df(x, n=5000):
        if len(x) > n:
            return x.sample(n=n, random_state=42)
        return x

    pos_df = sample_df(pos_df)
    neg_df = sample_df(neg_df)

    return pos_df, neg_df


# -------- 3. 统计词频并根据情感区分度筛选词汇 --------

def compute_class_counters(pos_df, neg_df):
    """分别统计正负样本中的词频"""
    pos_counter = Counter()
    neg_counter = Counter()

    for text in pos_df["text"]:
        tokens = preprocess_text(text)
        pos_counter.update(tokens)

    for text in neg_df["text"]:
        tokens = preprocess_text(text)
        neg_counter.update(tokens)

    return pos_counter, neg_counter


def build_sentiment_freqs(pos_counter, neg_counter, sentiment="positive",
                          min_total=30, min_ratio=0.7, min_class_count=15):
    """
    根据“在某一类中更偏向出现”的原则，挑出真正具有情感区分度的词。

    - sentiment: "positive" 或 "negative"
    - min_total: 该词在正负样本中总出现次数至少为 min_total
    - min_ratio: 该词在目标情感类别中的占比至少为 min_ratio
    - min_class_count: 该词在该类别中的出现次数至少为 min_class_count
    """
    freqs = {}

    all_words = set(pos_counter.keys()) | set(neg_counter.keys())
    for w in all_words:
        pos_c = pos_counter.get(w, 0)
        neg_c = neg_counter.get(w, 0)
        total = pos_c + neg_c
        if total < min_total:
            continue  # 太稀有的词跳过

        if sentiment == "positive":
            if pos_c < min_class_count:
                continue
            ratio = pos_c / (total + 1e-9)
            if ratio < min_ratio:
                continue
            freqs[w] = pos_c
        else:
            if neg_c < min_class_count:
                continue
            ratio = neg_c / (total + 1e-9)
            if ratio < min_ratio:
                continue
            freqs[w] = neg_c

    return freqs


def plot_wordcloud(freq_dict, title, out_file=None):
    """基于频率字典绘制词云，只展示筛选后的高区分度情感词"""
    if not freq_dict:
        print(f"⚠️ 词频字典为空，'{title}' 无法生成词云（筛选条件可能过严）。")
        return

    wc = WordCloud(
        width=1000,
        height=500,
        background_color="white",
        max_words=200
    ).generate_from_frequencies(freq_dict)

    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(title)
    plt.tight_layout()

    if out_file:
        plt.savefig(out_file, dpi=300)
        print(f"✅ 已保存词云到: {out_file}")

    plt.show()


def main():
    print("👉 正在加载并拆分正负样本...")
    pos_df, neg_df = load_data("test.csv")
    print(f"  Positive 样本数: {len(pos_df)}")
    print(f"  Negative 样本数: {len(neg_df)}")

    print("👉 统计正负样本词频...")
    pos_counter, neg_counter = compute_class_counters(pos_df, neg_df)

    print("👉 根据区分度筛选 Positive 情感词...")
    pos_freqs = build_sentiment_freqs(
        pos_counter,
        neg_counter,
        sentiment="positive",
        min_total=30,
        min_ratio=0.7,
        min_class_count=15,
    )

    print("👉 根据区分度筛选 Negative 情感词...")
    neg_freqs = build_sentiment_freqs(
        pos_counter,
        neg_counter,
        sentiment="negative",
        min_total=30,
        min_ratio=0.7,
        min_class_count=15,
    )

    print("👉 生成 Positive 词云...")
    plot_wordcloud(pos_freqs, title="Positive reviews word cloud (discriminative)", out_file="wordcloud_positive.png")

    print("👉 生成 Negative 词云...")
    plot_wordcloud(neg_freqs, title="Negative reviews word cloud (discriminative)", out_file="wordcloud_negative.png")


if __name__ == "__main__":
    main()