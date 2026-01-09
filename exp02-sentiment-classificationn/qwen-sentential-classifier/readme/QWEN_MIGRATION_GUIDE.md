# 从BERT迁移到Qwen模型指南

## 📋 快速检查清单

在修改代码之前，请按以下步骤操作：

### 1️⃣ 运行兼容性检查

```bash
cd qwen-sentential-classifier
python check_qwen_compatibility.py
```

这个脚本会：
- ✅ 测试Qwen模型是否能正常加载
- ✅ 检查tokenizer兼容性
- ✅ 验证模型输出格式
- ✅ 评估内存需求
- ✅ 给出推荐配置

---

## 🔧 需要修改的文件

### 文件1: config.py

**当前配置:**
```python
model_name = "bert-base-chinese"
max_seq_length = 64
batch_size = 16
```

**修改为（Qwen-1.8B 示例）:**
```python
model_name = "Qwen/Qwen-1_8B"  # 或 "Qwen/Qwen2-1.5B"
max_seq_length = 512  # Qwen支持更长序列
batch_size = 8  # 根据GPU显存调整
```

**修改为（Qwen-7B 示例）:**
```python
model_name = "Qwen/Qwen-7B"
max_seq_length = 512
batch_size = 4  # 7B模型更大，需要更小的batch_size
```

---

### 文件2: model.py

**需要修改的地方:**

在第21行，修改 `from_pretrained` 调用：

**修改前:**
```python
self.base_model = AutoModel.from_pretrained(model_name)
```

**修改后:**
```python
self.base_model = AutoModel.from_pretrained(
    model_name,
    trust_remote_code=True  # Qwen模型需要这个参数
)
```

**✅ 好消息**: model.py 的 forward 方法已经处理了 Qwen 没有 pooler_output 的情况（第49-53行），不需要额外修改！

---

### 文件3: main.py

**需要修改的地方:**

在第91行，修改 tokenizer 加载：

**修改前:**
```python
tokenizer = AutoTokenizer.from_pretrained(config.model_name)
```

**修改后:**
```python
tokenizer = AutoTokenizer.from_pretrained(
    config.model_name,
    trust_remote_code=True  # Qwen模型需要这个参数
)
```

在第177行（predict函数中），同样修改：

**修改前:**
```python
tokenizer = AutoTokenizer.from_pretrained(config.model_name)
```

**修改后:**
```python
tokenizer = AutoTokenizer.from_pretrained(
    config.model_name,
    trust_remote_code=True
)
```

---

### 文件4-8: 实验脚本（可选但推荐）

所有实验脚本中加载 tokenizer 和 model 的地方都需要添加 `trust_remote_code=True`：

**需要修改的文件:**
- compare_trained_untrained.py
- sample_stability_analysis.py
- train_size_analysis.py
- epoch_analysis.py

**在这些文件中找到并修改:**
```python
# 修改tokenizer加载
tokenizer = AutoTokenizer.from_pretrained(
    config.model_name,
    trust_remote_code=True
)

# 如果有直接加载AutoModel的地方，也要添加
model = AutoModel.from_pretrained(
    config.model_name,
    trust_remote_code=True
)
```

---

## 🎯 推荐的Qwen模型选择

| 模型 | 参数量 | 显存需求 | 推荐batch_size | 适用场景 |
|------|--------|---------|---------------|---------|
| Qwen/Qwen-1_8B | 1.8B | ~8GB | 8-16 | 资源受限、快速实验 |
| Qwen/Qwen2-1.5B | 1.5B | ~6GB | 16 | 最轻量级 |
| Qwen/Qwen-7B | 7B | ~28GB | 2-4 | 追求性能 |
| Qwen/Qwen2-7B | 7B | ~28GB | 2-4 | Qwen2系列，更新 |

**建议**:
- **如果是学习/实验**: 使用 Qwen-1.8B 或 Qwen2-1.5B
- **如果追求性能**: 使用 Qwen-7B（需要较好的GPU）

---

## ⚙️ 完整修改步骤

### 步骤1: 检查兼容性

```bash
python check_qwen_compatibility.py
```

选择你想使用的Qwen模型，查看检查结果。

### 步骤2: 修改config.py

```bash
# 直接修改，或使用下面的命令
nano config.py  # 或 vim/vscode
```

修改 `model_name` 为你选择的Qwen模型。

### 步骤3: 批量修改其他文件

我可以为你创建一个自动修改脚本，或者你可以手动修改每个文件中的 `from_pretrained` 调用。

### 步骤4: 测试运行

```bash
# 先用小样本测试
python main.py
```

观察是否有错误。

### 步骤5: 运行实验

```bash
./run_all_experiments.sh
```

---

## ❗ 常见问题和解决方案

### 问题1: ImportError: trust_remote_code

**错误信息:**
```
ValueError: ... requires you to execute the modeling file ... set `trust_remote_code=True`
```

**解决方案:**
确保所有 `from_pretrained` 都添加了 `trust_remote_code=True`

---

### 问题2: 显存不足 (CUDA out of memory)

**错误信息:**
```
RuntimeError: CUDA out of memory
```

**解决方案:**
1. 减小 `batch_size` (如从16改为4)
2. 减小 `max_seq_length` (如从512改为256)
3. 使用更小的Qwen模型
4. 使用梯度累积：
   ```python
   accumulation_steps = 4  # 在训练循环中添加
   ```

---

### 问题3: 模型下载慢

**解决方案:**
代码已经设置了镜像：
```python
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
```

如果还是慢，可以手动下载模型：
```bash
# 使用huggingface-cli
pip install huggingface_hub
huggingface-cli download Qwen/Qwen-1_8B --local-dir ./models/qwen-1.8b
```

然后修改 config.py:
```python
model_name = "./models/qwen-1.8b"
```

---

### 问题4: 模型输出维度不匹配

**错误信息:**
```
RuntimeError: mat1 and mat2 shapes cannot be multiplied
```

**解决方案:**
这不应该发生，因为分类器会根据 `model.config.hidden_size` 自动适配。

如果出现，检查：
1. 是否正确加载了模型
2. config.py 中的 model_name 是否正确

---

## 🔍 验证修改是否正确

运行这个简单测试：

```python
from config import Config
from transformers import AutoTokenizer, AutoModel

config = Config()

# 测试tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    config.model_name,
    trust_remote_code=True
)
print(f"✅ Tokenizer加载成功")

# 测试model
model = AutoModel.from_pretrained(
    config.model_name,
    trust_remote_code=True
)
print(f"✅ Model加载成功")
print(f"Hidden size: {model.config.hidden_size}")
```

如果都成功，说明配置正确！

---

## 📊 性能对比预期

从BERT-base-chinese迁移到Qwen后，你可能会看到：

| 指标 | BERT-base-chinese | Qwen-1.8B | Qwen-7B |
|------|-------------------|-----------|---------|
| 准确率 | ~85-88% | ~88-90% | ~90-92% |
| 训练时间/epoch | 1x | 1.2-1.5x | 2-3x |
| 显存占用 | ~4GB | ~8GB | ~28GB |
| 推理速度 | 1x | 0.8x | 0.3x |

**注**: 具体数值取决于数据集和硬件

---

## 📝 下一步

1. ✅ 运行 `check_qwen_compatibility.py` 检查兼容性
2. ✅ 根据建议修改 config.py
3. ✅ 修改 model.py 和 main.py 添加 `trust_remote_code=True`
4. ✅ 可选：修改实验脚本
5. ✅ 运行 `python main.py` 测试训练
6. ✅ 运行实验脚本验证

---

**需要帮助？**
- 运行兼容性检查脚本会给出详细建议
- 如果遇到错误，查看"常见问题"部分
- 可以先在小数据集上测试（修改 config.py 的 max_train_samples）
