# TextCNN Sentiment Classification

基于 TextCNN（Kim, 2014）的情感分类模型，用于对产品评论进行正面/负面情感分析。

## 📋 项目结构

```
textcnn-sentiment-classifier/
├── config.py              # 配置文件
├── data_loader.py         # 数据加载和预处理
├── model.py              # TextCNN模型定义
├── train.py              # 训练脚本
├── predict.py            # 预测脚本
├── README.md             # 项目说明
├── requirements.txt      # 依赖包
├── outputs/              # 输出目录
│   ├── textcnn_model.pth    # 训练好的模型
│   ├── vocab.pkl            # 词表
│   ├── training_curves.png  # 训练曲线
│   └── test_results.json    # 测试结果
└── logs/                 # 日志目录
    └── training_history.json
```

## 🚀 快速开始

### 1. 环境准备

```bash
cd textcnn-sentiment-classifier

# 创建虚拟环境（可选）
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 训练模型

```bash
python train.py
```

训练过程会：
- 自动加载数据（train.csv, dev.csv, test.csv）
- 构建词表并保存
- 训练 TextCNN 模型
- 在验证集上评估并保存最佳模型
- 在测试集上进行最终评估
- 生成训练曲线图

### 3. 使用模型预测

```bash
python predict.py
```

这将启动交互式预测界面，您可以输入评论文本进行情感分析。

## 🔧 配置说明

在 `config.py` 中可以调整以下参数：

### 数据配置
- `MAX_VOCAB_SIZE`: 词表最大大小（默认：50000）
- `MIN_WORD_FREQ`: 最小词频（默认：2）
- `MAX_SEQ_LENGTH`: 最大序列长度（默认：256）

### 模型配置
- `EMBEDDING_DIM`: 词向量维度（默认：300）
- `NUM_FILTERS`: 每个卷积核的数量（默认：100）
- `FILTER_SIZES`: 卷积核大小列表（默认：[3, 4, 5]）
- `DROPOUT_RATE`: Dropout 比率（默认：0.5）

### 训练配置
- `BATCH_SIZE`: 批次大小（默认：64）
- `NUM_EPOCHS`: 训练轮数（默认：10）
- `LEARNING_RATE`: 学习率（默认：0.001）
- `EARLY_STOPPING`: 是否使用早停（默认：True）
- `PATIENCE`: 早停的耐心值（默认：3）

## 📊 模型架构

TextCNN 模型结构：

```
输入文本
    ↓
Embedding层 (vocab_size × embedding_dim)
    ↓
Conv2D层（多个不同大小的卷积核）
    ├─ kernel_size=3 → num_filters个特征图
    ├─ kernel_size=4 → num_filters个特征图
    └─ kernel_size=5 → num_filters个特征图
    ↓
Max Pooling (每个特征图取最大值)
    ↓
拼接所有特征
    ↓
Dropout (防止过拟合)
    ↓
全连接层 (分类)
    ↓
输出 (2类：负面/正面)
```

**特点：**
- 多尺度卷积核（3-gram, 4-gram, 5-gram）捕获不同长度的局部特征
- Max pooling 提取最显著特征
- 简单高效，训练速度快

## 📈 实验结果

训练完成后，会生成以下文件：

1. **模型权重**: `outputs/textcnn_model.pth`
2. **词表**: `outputs/vocab.pkl`
3. **训练曲线**: `outputs/training_curves.png`
4. **测试结果**: `outputs/test_results.json`
5. **训练历史**: `logs/training_history.json`

## 🔬 使用示例

### Python API 使用

```python
from predict import SentimentPredictor
import config

# 初始化预测器
predictor = SentimentPredictor(
    model_path=config.MODEL_SAVE_PATH,
    vocab_path=config.VOCAB_SAVE_PATH
)

# 单个预测
text = "This product is amazing! I love it."
pred, confidence = predictor.predict(text)
print(f"Prediction: {predictor.label_names[pred]} (confidence: {confidence:.4f})")

# 批量预测
texts = [
    "Great quality!",
    "Terrible, don't buy.",
    "It's okay."
]
predictions, confidences = predictor.predict_batch(texts)
for text, pred, conf in zip(texts, predictions, confidences):
    print(f"{text} → {predictor.label_names[pred]} ({conf:.4f})")
```

### 测试单个模块

```bash
# 测试数据加载
python data_loader.py

# 测试模型
python model.py
```

## 📚 参考文献

- Kim, Y. (2014). Convolutional Neural Networks for Sentence Classification. EMNLP 2014.
- 论文链接: https://arxiv.org/abs/1408.5882

## ⚙️ 技术栈

- **PyTorch**: 深度学习框架
- **NumPy**: 数值计算
- **scikit-learn**: 评估指标
- **matplotlib**: 可视化
- **tqdm**: 进度条

## 🐛 常见问题

### 1. CUDA 内存不足

如果遇到 GPU 内存不足，可以：
- 减小 `BATCH_SIZE`（如改为 32 或 16）
- 减小 `MAX_SEQ_LENGTH`（如改为 128）
- 减小 `NUM_FILTERS`（如改为 50）

### 2. 训练速度慢

如果训练速度慢，可以：
- 增大 `BATCH_SIZE`（如果显存允许）
- 设置 `MAX_TRAIN_SAMPLES` 限制训练样本数
- 使用 GPU 训练（设置 `DEVICE="cuda"`）

### 3. 准确率不理想

如果准确率不理想，可以尝试：
- 增大 `NUM_FILTERS`（如改为 200）
- 调整 `FILTER_SIZES`（如 [2, 3, 4, 5]）
- 增大 `EMBEDDING_DIM`（如改为 512）
- 降低 `DROPOUT_RATE`（如改为 0.3）
- 增加训练轮数 `NUM_EPOCHS`

## 📝 数据格式

训练数据格式（CSV）：
```
label,title,text
1,"Bad product","This is terrible..."
2,"Great!","I love this product..."
```

- `label`: 1=负面，2=正面
- `title`: 评论标题
- `text`: 评论正文

## 🎯 性能优化建议

1. **使用预训练词向量**: 
   - 可以使用 Word2Vec、GloVe 等预训练词向量
   - 在 `config.py` 中设置 `WORD2VEC_PATH`

2. **数据增强**:
   - 同义词替换
   - 回译（back-translation）
   - 随机删除/插入

3. **模型改进**:
   - 尝试 Multi-channel TextCNN
   - 添加 Attention 机制
   - 使用双向 LSTM + CNN

## 📧 联系方式

如有问题，请提交 Issue 或联系开发者。

---

**Happy Training! 🚀**

