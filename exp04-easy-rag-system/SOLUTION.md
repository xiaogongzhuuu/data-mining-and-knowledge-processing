# 医疗 RAG 系统 - ChromaDB 离线模式完整解决方案

## 📋 问题分析

### 原始问题

```
❌ Failed to create or get collection: We couldn't connect to 'https://huggingface.co'
to load the files, and couldn't find them in the cached files.
```

### 根本原因

ChromaDB 在创建 collection 时，如果不指定自定义的嵌入函数，会自动尝试下载默认的嵌入模型（通常是 `all-MiniLM-L6-v2`）。即使设置了离线模式的环境变量，这个自动下载的流程仍然会被触发。

### 关键问题点

1. **自动模型下载**: ChromaDB 的 `create_collection()` 方法默认行为
2. **网络连接尝试**: 离线环境变量不能完全阻止 ChromaDB 的联网尝试
3. **版本兼容性**: 不同版本的 ChromaDB 有不同的 API 和行为

---

## ✅ 解决方案

### 核心策略：使用虚拟嵌入函数

替代原来的做法，我们使用一个 **虚拟嵌入函数** 来阻止 ChromaDB 自动下载模型。这个虚拟函数返回预定义维度的零向量，实际的嵌入向量在索引和搜索时由我们的嵌入模型预先计算。

### 关键代码改动

#### 1. 定义虚拟嵌入函数

```python
class DummyEmbeddingFunction(embedding_functions.EmbeddingFunction):
    """
    虚拟嵌入函数 - 这只是为了让 ChromaDB 不尝试下载默认模型。
    实际的嵌入向量会在 index_data_if_needed 中预先计算。
    """
    def __call__(self, input):
        if isinstance(input, str):
            input = [input]
        # 返回虚拟向量列表（全零）
        return [[0.0] * EMBEDDING_DIM for _ in input]
```

#### 2. 创建 Collection 时使用虚拟函数

```python
def setup_chroma_collection(_client):
    # ...
    dummy_embedding_fn = DummyEmbeddingFunction()
    collection = _client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
        embedding_function=dummy_embedding_fn,  # 关键：提供自定义函数
        get_or_create=False
    )
```

#### 3. 索引时提供预计算的嵌入

```python
def index_data_if_needed(client, data, embedding_model):
    # ...
    embeddings = embedding_model.encode(docs_for_embedding, show_progress_bar=True)
    collection.add(
        ids=ids,
        embeddings=embeddings.tolist(),  # 提供预计算的向量
        documents=docs_for_embedding,
        metadatas=metadatas
    )
```

---

## 🚀 快速开始

### 前置条件

```bash
pip3 install chromadb sentence-transformers streamlit requests
```

### 步骤 1：清理旧数据

```bash
rm -rf ./chroma_data
```

### 步骤 2：运行离线测试（验证修复）

```bash
python3 test_chroma_offline.py
```

预期输出：

```
============================================================
🧪 离线模式 ChromaDB 测试
============================================================

[步骤 1] 初始化 ChromaDB 客户端...
✅ 使用 PersistentClient

[步骤 2] 清理旧的 collection...
✅ 已删除旧的 collection

[步骤 3] 创建 collection (使用虚拟嵌入函数)...
✅ Collection 'medical_rag_lite' 创建成功！

[步骤 4] 加载嵌入模型...
✅ 嵌入模型加载成功

[步骤 5] 测试数据索引...
✅ 成功索引 3 个文档

[步骤 6] 测试搜索功能...
✅ 搜索功能正常

============================================================
✅ 所有测试通过！离线模式工作正常
============================================================
```

### 步骤 3：启动应用

#### 选项 A：使用完整启动脚本（推荐）

```bash
chmod +x start_complete.sh
./start_complete.sh
```

#### 选项 B：直接启动 Streamlit

```bash
streamlit run app.py
```

### 步骤 4：（可选）启用生成功能

在另一个终端启动 Ollama：

```bash
ollama serve
```

然后确保 `qwen3:8b` 已下载：

```bash
ollama pull qwen3:8b
```

---

## 🔍 工作原理详解

### 数据流

```
用户查询
    ↓
app.py 调用 hybrid_search()
    ↓
retrieval_optimizer.py 中的 search_similar_documents()
    ↓
chroma_utils.py 中的 get_collection()
    ↓
使用虚拟嵌入函数获取 collection（不下载任何模型）
    ↓
使用本地加载的嵌入模型生成查询向量
    ↓
ChromaDB 执行向量相似度搜索
    ↓
返回相关文档
    ↓
（可选）Ollama 生成答案
```

### 为什么虚拟函数有效

1. **阻止自动下载**: ChromaDB 看到有提供的嵌入函数，就不会尝试下载默认模型
2. **不会被调用**: 虚拟函数返回的向量不会被使用，因为我们总是提供预计算的向量
3. **完全离线**: 整个流程都在本地完成，无需网络连接

---

## 📊 文件修改总结

### 修改的文件

1. **chroma_utils.py**
   - 添加 `DummyEmbeddingFunction` 类
   - 修改 `setup_chroma_collection()` 使用虚拟函数
   - 修改 `index_data_if_needed()` 使用虚拟函数
   - 修改 `search_similar_documents()` 使用虚拟函数

### 新增的文件

1. **test_chroma_offline.py** - 完整的离线测试套件
2. **start_complete.sh** - 完整的启动脚本
3. **SOLUTION.md** - 本文档

---

## 🐛 故障排除

### 问题 1：仍然尝试连接 HuggingFace

**症状**: 看到 "couldn't connect to huggingface.co" 错误

**解决方案**:

1. 确保已运行 `rm -rf ./chroma_data` 清理旧数据
2. 确保 `chroma_utils.py` 已更新，使用 `DummyEmbeddingFunction`
3. 运行 `python3 test_chroma_offline.py` 验证修复

### 问题 2：嵌入模型加载失败

**症状**: "Failed to load embedding model" 错误

**解决方案**:

1. 检查 `./hf_cache` 目录是否存在且包含模型文件
2. 确保 `EMBEDDING_MODEL_NAME` 在 config.py 中正确配置
3. 尝试在线模式临时加载模型（如果可以），然后再切换回离线模式

### 问题 3：collection 创建成功但搜索失败

**症状**: 创建成功，但搜索时出错

**解决方案**:

1. 确保数据已正确索引（检查 `collection.count()` > 0）
2. 检查嵌入向量维度是否匹配（应为 768）
3. 查看 Streamlit 的详细日志了解具体错误

### 问题 4：Ollama 连接失败

**症状**: "Cannot connect to Ollama service" 警告

**解决方案**:
这是正常的。系统会自动进入 **搜索模式**，仅执行文档检索，不生成答案。
如需启用答案生成，运行：

```bash
ollama serve
ollama pull qwen3:8b
```

---

## 🔧 高级配置

### 修改嵌入模型

编辑 `config.py`:

```python
EMBEDDING_MODEL_NAME = 'your-model-name'  # 改为其他 sentence-transformer 模型
EMBEDDING_DIM = 384  # 更新维度（例如，如果使用其他模型）
```

### 修改 Ollama 配置

编辑 `config.py`:

```python
OLLAMA_BASE_URL = "http://localhost:11434"  # 改为您的 Ollama 地址
OLLAMA_MODEL = "qwen3:8b"  # 改为其他模型
```

### 修改搜索参数

编辑 `config.py`:

```python
TOP_K = 5  # 返回前 K 个结果
MAX_ARTICLES_TO_INDEX = 500  # 最多索引多少篇文章
```

---

## 📈 性能优化建议

1. **向量维度**: 768 维向量可能对某些应用过大，可考虑使用 384 维模型（如 `all-MiniLM-L6-v2`）以加快搜索
2. **索引大小**: 根据硬件和响应时间要求调整 `MAX_ARTICLES_TO_INDEX`
3. **Top K**: 增加 `TOP_K` 以提高召回率，但会增加生成时间

---

## ✨ 总结

这个解决方案通过以下方式完全解决了离线模式的问题：

✅ **无需网络连接** - 所有模型和数据都在本地处理
✅ **完全离线** - ChromaDB 不再尝试自动下载任何模型
✅ **灵活可靠** - 虚拟嵌入函数简单有效
✅ **易于部署** - 只需修改 `chroma_utils.py` 和清理旧数据

现在您可以在完全离线的环境中运行医疗 RAG 系统了！🎉
