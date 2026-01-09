# 使用Qwen模型 - 快速指南

## ✅ 接口兼容性检查结果

我已经检查了你的代码，**好消息**：你的代码架构已经支持Qwen模型！

### 为什么兼容？

1. ✅ **model.py (第21行)**: 使用 `AutoModel.from_pretrained()`，支持任意Transformer模型
2. ✅ **model.py (第49-53行)**: 已经处理了Qwen没有`pooler_output`的情况
3. ✅ **main.py (第91行)**: 使用 `AutoTokenizer.from_pretrained()`，支持任意tokenizer
4. ✅ **main.py (第94-95行)**: 已经处理了`pad_token`的问题

### 需要修改什么？

**唯一需要添加的是 `trust_remote_code=True` 参数**，因为Qwen模型使用了自定义代码。

---

## 🚀 三种使用方式

### 方式1: 自动修改（推荐⭐）

```bash
cd qwen-sentential-classifier

# 步骤1: 检查Qwen模型兼容性
python check_qwen_compatibility.py

# 步骤2: 自动添加trust_remote_code参数
python auto_add_trust_remote_code.py

# 步骤3: 修改config.py中的model_name
# 打开config.py，将第6行改为：
# model_name = "Qwen/Qwen-1_8B"  # 或其他Qwen模型

# 步骤4: 测试运行
python main.py
```

### 方式2: 手动修改

按照 `QWEN_MIGRATION_GUIDE.md` 中的说明，手动修改以下文件：
- config.py (修改model_name)
- model.py (添加trust_remote_code=True)
- main.py (添加trust_remote_code=True)
- 实验脚本 (可选)

### 方式3: 仅修改config.py（快速测试）

如果只想快速测试，可以：

1. 修改 config.py:
```python
model_name = "Qwen/Qwen-1_8B"
```

2. 临时修改 model.py 第21行:
```python
self.base_model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
```

3. 临时修改 main.py 第91行:
```python
tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
```

---

## 📊 推荐配置

### 如果你的GPU显存 >= 16GB
```python
# config.py
model_name = "Qwen/Qwen-1_8B"
max_seq_length = 512
batch_size = 8
```

### 如果你的GPU显存 >= 32GB
```python
# config.py
model_name = "Qwen/Qwen-7B"
max_seq_length = 512
batch_size = 4
```

### 如果只有CPU或显存 < 8GB
```python
# config.py
model_name = "Qwen/Qwen2-1.5B"  # 最小的Qwen模型
max_seq_length = 256
batch_size = 4
```

---

## 🔍 验证步骤

### 1. 运行兼容性检查
```bash
python check_qwen_compatibility.py
```

看到这些输出说明成功：
```
✅ Tokenizer加载成功
✅ Model加载成功
✅ 分类器输出shape: torch.Size([1, 2])
✅ 兼容性检查通过！
```

### 2. 快速测试
```python
from config import Config
from model import SentimentClassifier

config = Config()
model = SentimentClassifier(config.model_name, config.num_classes)
print("✅ 模型创建成功！")
```

---

## ⚠️ 可能遇到的问题

### 问题1: 需要 trust_remote_code=True

**错误信息:**
```
ValueError: ... requires you to execute code in that repo ... set `trust_remote_code=True`
```

**解决方案:**
运行自动修改脚本：
```bash
python auto_add_trust_remote_code.py
```

### 问题2: CUDA Out of Memory

**错误信息:**
```
RuntimeError: CUDA out of memory
```

**解决方案:**
在 config.py 中减小参数：
```python
batch_size = 4  # 或更小
max_seq_length = 256  # 或更小
```

### 问题3: 下载速度慢

**解决方案:**
代码已设置镜像（main.py第23行），应该会从国内镜像下载。

如果还是慢，可以手动下载：
```bash
git clone https://www.modelscope.cn/qwen/Qwen-1_8B.git
```

然后修改 config.py:
```python
model_name = "./Qwen-1_8B"  # 本地路径
```

---

## 📋 修改前后对比

### Before (BERT):
```python
# config.py
model_name = "bert-base-chinese"
max_seq_length = 64
batch_size = 16

# model.py
self.base_model = AutoModel.from_pretrained(model_name)

# main.py
tokenizer = AutoTokenizer.from_pretrained(config.model_name)
```

### After (Qwen):
```python
# config.py
model_name = "Qwen/Qwen-1_8B"
max_seq_length = 512
batch_size = 8

# model.py
self.base_model = AutoModel.from_pretrained(
    model_name,
    trust_remote_code=True
)

# main.py
tokenizer = AutoTokenizer.from_pretrained(
    config.model_name,
    trust_remote_code=True
)
```

---

## 🎯 总结

**你的代码已经很好了！** 只需要：

1. ✅ 添加 `trust_remote_code=True` 参数（3处）
2. ✅ 修改 config.py 中的 model_name
3. ✅ 可选：调整 batch_size 和 max_seq_length

**最简单的方法：**
```bash
# 一键自动配置
python auto_add_trust_remote_code.py

# 然后手动修改 config.py 的 model_name
# 完成！
```

---

## 📞 需要帮助？

查看详细文档：
- `QWEN_MIGRATION_GUIDE.md` - 完整迁移指南
- `check_qwen_compatibility.py` - 兼容性检查工具
- `auto_add_trust_remote_code.py` - 自动修改工具
