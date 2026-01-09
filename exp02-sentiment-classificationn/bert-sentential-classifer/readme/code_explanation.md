# BERT 情感分类项目代码详解与参数调整说明

## 1. 参数调整说明 (针对低配设备)

为了让项目在您的设备上顺利运行，我对 `config.py` 和 `main.py` 进行了以下关键调整。这些调整旨在降低显存占用和计算时间，同时保持代码的核心逻辑不变。

### 1.1 降低序列长度 (`max_seq_length`)
*   **原值**: 128
*   **新值**: **64**
*   **原因**: BERT 的计算复杂度与序列长度的平方成正比。将长度减半，显存占用和计算量会大幅下降。对于情感分类任务，64 个 token 通常足以覆盖大部分句子的关键信息。

### 1.2 减小批次大小 (`batch_size`)
*   **原值**: 16
*   **新值**: **8**
*   **原因**: 批次大小直接决定了显存的峰值占用。如果您的显存较小（如 4GB 或更少），设置为 8 是比较安全的起点。

### 1.3 限制样本数量 (`max_train_samples`)
*   **原值**: 20000 (硬编码)
*   **新值**: **2000** (在 Config 中配置)
*   **原因**: 完整的训练集可能非常大，导致训练一个 epoch 需要数小时。限制样本数可以让您在几分钟内跑通整个流程，验证代码无误。
*   **注意**: 验证集样本数也限制到了 **500**。

### 1.4 关于“词向量维度”
您提到的“词向量维度”在 BERT 中通常由预训练模型决定（例如 `bert-base` 是 768 维）。
*   **当前状态**: 保持使用 `bert-base-chinese` (768维)。
*   **进一步优化**: 如果上述调整后依然跑不动，建议更换更小的预训练模型，例如 `hfl/rbt3` (3层 BERT，维度依然是 768 但层数少) 或 `ckiplab/bert-tiny-chinese` (维度 312)。

---

## 2. 整体代码逻辑详解

项目遵循标准的 PyTorch 深度学习流程：**数据加载 -> 预处理 -> 模型构建 -> 训练 -> 评估**。

### 2.1 数据加载 (`load_data.py`)
*   负责读取原始的 CSV 文件 (`train.csv`, `dev.csv`, `test.csv`)。
*   将数据解析为两个列表：`texts` (文本内容) 和 `labels` (情感标签，0或1)。

### 2.2 数据集构建 (`dataset.py`)
这是连接原始文本和 BERT 模型的桥梁。
*   **`SentimentDataset` 类**: 继承自 `torch.utils.data.Dataset`。
*   **核心方法 `__getitem__`**:
    1.  接收一个文本样本。
    2.  调用 `tokenizer.encode_plus`:
        *   **Tokenization**: 将文本切分为字/词。
        *   **Mapping**: 将字转换为 ID。
        *   **Padding/Truncation**: 统一填充或截断到 `max_seq_length` (64)。
        *   **Special Tokens**: 添加 `[CLS]` (句首) 和 `[SEP]` (句尾)。
    3.  返回 `input_ids` (ID序列), `attention_mask` (标记哪些是真实字，哪些是填充), `labels`。

### 2.3 模型结构 (`model.py`)
*   **`SentimentClassifier` 类**:
    *   **BERT 层**: `BertModel.from_pretrained(...)`。这是核心部分，负责将输入的 ID 序列转换为高维向量表示。
    *   **Dropout 层**: 防止过拟合。
    *   **分类层**: `nn.Linear(768, 2)`。将 BERT 输出的 `[CLS]` 向量（代表整句语义，768维）映射到 2 个类别上。

### 2.4 训练主流程 (`main.py`)
1.  **初始化**: 加载 Config，设置 Device (CPU/GPU)，初始化 Tokenizer 和 Model。
2.  **数据准备**:
    *   使用 `DataLoader` 将 Dataset 封装为可迭代的 batch。
    *   在此处应用了我们新增的样本截断逻辑。
3.  **训练循环 (`train` 函数)**:
    *   **Forward**: 输入数据 -> 模型 -> 得到预测结果。
    *   **Loss**: 计算预测结果与真实标签的交叉熵损失 (`CrossEntropyLoss`)。
    *   **Backward**: 反向传播计算梯度。
    *   **Optimizer**: `AdamW` 更新模型参数。
4.  **评估与保存**:
    *   每个 Epoch 结束后，在验证集上计算 Loss 和 Accuracy。
    *   如果当前 Accuracy 优于历史最佳，则保存模型权重 (`save_model`)。

---

## 3. 下一步建议

1.  **运行测试**: 直接运行 `main.py` 或 `run.sh`。
2.  **观察资源**: 观察运行时的显存/内存占用和速度。
3.  **逐步恢复**: 如果运行顺畅且速度很快，可以尝试在 `config.py` 中逐步增加 `batch_size` (8 -> 16) 或 `max_train_samples` (2000 -> 5000 -> 10000)，以获得更好的模型效果。
