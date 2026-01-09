"""
配置文件：TextCNN 情感分类
"""

import os

# ==================== 路径配置 ====================
# 数据路径（相对于项目根目录）
DATA_DIR = "../bert-sentential-classifer"
TRAIN_FILE = os.path.join(DATA_DIR, "train.csv")
DEV_FILE = os.path.join(DATA_DIR, "dev.csv")
TEST_FILE = os.path.join(DATA_DIR, "test.csv")

# 输出路径
OUTPUT_DIR = "./outputs"
MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "textcnn_model.pth")
VOCAB_SAVE_PATH = os.path.join(OUTPUT_DIR, "vocab.pkl")
LOG_DIR = "./logs"

# Word2Vec预训练模型路径（可选）
WORD2VEC_PATH = None  # 如果有预训练的word2vec，填入路径，否则None

# ==================== 数据预处理配置 ====================
MAX_VOCAB_SIZE = 50000      # 词表最大大小
MIN_WORD_FREQ = 2           # 最小词频
MAX_SEQ_LENGTH = 256        # 最大序列长度
PADDING_TOKEN = "<PAD>"     # padding标记
UNKNOWN_TOKEN = "<UNK>"     # 未知词标记

# ==================== 模型配置 ====================
EMBEDDING_DIM = 300         # 词向量维度
NUM_FILTERS = 100           # 每个卷积核的数量
FILTER_SIZES = [3, 4, 5]    # 卷积核大小（窗口大小）
DROPOUT_RATE = 0.5          # Dropout比率
NUM_CLASSES = 2             # 分类数量（正面、负面）

# ==================== 训练配置 ====================
BATCH_SIZE = 64             # 批次大小
NUM_EPOCHS = 10            # 训练轮数（增加到15轮）
LEARNING_RATE = 0.001       # 学习率
WEIGHT_DECAY = 1e-4         # 权重衰减

# 早停配置
EARLY_STOPPING = True       # 是否使用早停
PATIENCE = 3                # 早停的耐心值（多少个epoch不提升就停止）

# 设备配置
DEVICE = "cuda"             # cuda / cpu

# 随机种子
SEED = 42

# ==================== 实验配置 ====================
# 是否使用预训练的word2vec
USE_PRETRAINED_EMBEDDING = False if WORD2VEC_PATH is None else True

# 是否冻结embedding层
FREEZE_EMBEDDING = False

# 是否显示详细训练信息
VERBOSE = True

# 最大训练样本数（用于快速实验，None表示使用全部数据）
MAX_TRAIN_SAMPLES = 10000    # 使用10000条样本进行训练
BALANCE_TRAIN_DATA = True    # 是否均衡正负样本

# ==================== 创建必要的目录 ====================
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

