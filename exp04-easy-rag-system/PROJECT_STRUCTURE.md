# RAG 系统项目结构说明

## 一、项目整体架构

```
exp04-easy-rag-system/
├── 核心应用层
│   ├── app.py                    # Streamlit Web 应用主入口
│   ├── rag_core.py               # RAG 核心逻辑（答案生成）
│   └── run.sh                    # 启动脚本
│
├── 数据处理层
│   ├── data_utils.py             # 数据加载工具
│   ├── preprocess.py             # 数据预处理（文本分块）
│   └── chunk_optimizer.py        # 分块优化模块
│
├── 向量存储层
│   ├── chroma_utils.py           # ChromaDB 工具（正在使用）
│   ├── milvus_utils.py           # Milvus 工具（已废弃）
│   └── chroma.sqlite3            # ChromaDB 数据库文件
│
├── 模型层
│   ├── models.py                 # 模型加载（嵌入模型、生成模型）
│   └── retrieval_optimizer.py    # 检索优化（混合检索、重排序）
│
├── 配置层
│   ├── config.py                 # 系统配置参数
│   ├── config.toml               # Streamlit 配置
│   └── requirements.txt          # Python 依赖包
│
├── 测试与评估层
│   ├── test_models.py            # 模型测试
│   ├── test_search.py            # 检索测试
│   ├── test_rag_performance.py   # RAG 性能测试
│   ├── test_distance.py          # 距离计算测试
│   ├── evaluate_rag.py           # RAG 评估脚本
│   ├── diagnostics.py            # 系统诊断
│   ├── quick_validation.py       # 快速验证
│   ├── check_db.py               # 数据库检查
│   └── analyze_data.py           # 数据分析
│
├── 数据目录
│   ├── data/
│   │   └── processed_data.json   # 预处理后的数据
│   ├── sources/                  # 原始数据源（9个txt文件）
│   │   ├── 吴银根.txt
│   │   ├── 邵长荣.txt
│   │   ├── 施杞.txt
│   │   ├── 唐汉钧.txt
│   │   ├── 王大增.txt
│   │   ├── 王育群.txt
│   │   ├── 吴银根.txt
│   │   ├── 徐蓉娟.txt
│   │   └── 徐振晔.txt
│   └── hf_cache/                 # HuggingFace 模型缓存
│       └── hub/
│           └── models--moka-ai--m3e-base/
│
├── 文档目录
│   ├── QUICKSTART.md             # 快速启动指南
│   ├── SYSTEM_STATUS.md          # 系统状态文档
│   ├── OPTIMIZATION_SUMMARY.md   # 优化总结
│   ├── OPTIMIZATION_PLAN.md      # 优化计划
│   ├── BASELINE_METRICS.md       # 基线指标
│   ├── RAG_VALIDATION_GUIDE.md   # RAG 验证指南
│   ├── TROUBLESHOOTING.md        # 问题排查
│   ├── DIAGNOSTICS.md            # 诊断方案
│   ├── STARTUP_CHECKLIST.md      # 启动检查清单
│   ├── VALIDATION_GUIDE.md       # 验证指南
│   └── CHANGELOG.md              # 变更日志
│
└── 其他
    ├── __pycache__/              # Python 缓存
    ├── .streamlit/               # Streamlit 配置
    └── [4个向量索引目录]         # ChromaDB 索引文件
```

---

## 二、核心文件详解

### 2.1 应用入口层

#### `app.py` - Streamlit Web 应用主入口
**作用**：系统的前端界面和主控制器

**主要功能**：
- 初始化 ChromaDB 客户端
- 加载嵌入模型和生成模型
- 加载和索引数据
- 处理用户查询
- 显示检索结果和生成答案
- 显示系统配置信息

**依赖关系**：
```
app.py
├── config.py (配置参数)
├── data_utils.py (数据加载)
├── models.py (模型加载)
├── chroma_utils.py (向量数据库)
├── rag_core.py (答案生成)
└── retrieval_optimizer.py (检索优化)
```

**关键代码**：
```python
# 导入依赖
from config import DATA_FILE, EMBEDDING_MODEL_NAME, ...
from data_utils import load_data
from models import load_embedding_model, load_generation_model
from chroma_utils import get_chroma_client, setup_chroma_collection, ...
from rag_core import generate_answer
from retrieval_optimizer import hybrid_search, rerank_documents, ...

# 主流程
1. 初始化 ChromaDB
2. 加载模型
3. 加载数据
4. 索引数据（如果需要）
5. 处理用户查询
6. 显示结果
```

---

### 2.2 RAG 核心逻辑层

#### `rag_core.py` - RAG 核心逻辑
**作用**：生成答案的核心模块

**主要功能**：
- 构建上下文（从检索到的文档中提取信息）
- 设计提示词模板
- 调用 Ollama API 生成答案
- 处理生成错误（超时、连接错误等）

**依赖关系**：
```
rag_core.py
├── config.py (生成参数)
└── requests (HTTP 请求)
```

**关键代码**：
```python
def generate_answer(query, context_docs, gen_model, tokenizer):
    # 1. 构建上下文
    context_parts = []
    for doc in context_docs:
        context_parts.append(f"【文档】标题：{doc['title']}\n内容：{doc['content']}")
    
    # 2. 设计提示词
    prompt = f"""你是一位专业的中医医疗问答助手...
    【上下文文档】
    {context}
    【用户问题】
    {query}
    """
    
    # 3. 调用 Ollama API
    response = requests.post(f"{OLLAMA_BASE_URL}/api/generate", ...)
    
    # 4. 返回答案
    return answer
```

---

### 2.3 数据处理层

#### `data_utils.py` - 数据加载工具
**作用**：从 JSON 文件加载数据

**主要功能**：
- 读取 `processed_data.json`
- 返回数据列表

**依赖关系**：
```
data_utils.py
└── json (标准库)
```

**关键代码**：
```python
def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)
```

---

#### `preprocess.py` - 数据预处理
**作用**：将原始 txt 文件转换为 JSON 格式

**主要功能**：
- 从 `sources/` 目录读取 txt 文件
- 提取标题和正文
- 智能分块（512 字符，50 字符重叠）
- 生成 JSON 格式数据
- 保存到 `data/processed_data.json`

**依赖关系**：
```
preprocess.py
├── os (标准库)
├── json (标准库)
└── re (标准库)
```

**关键代码**：
```python
# 主流程
1. 遍历 sources/ 目录
2. 读取每个 txt 文件
3. 提取标题和正文
4. 使用 split_text() 分块
5. 生成 JSON 数据
6. 保存到 processed_data.json
```

---

#### `chunk_optimizer.py` - 分块优化
**作用**：优化数据分块策略

**主要功能**：
- 智能文本分块（在语义边界切分）
- 优化数据分块策略
- 增强文档元数据
- 文档关键词提取

**依赖关系**：
```
chunk_optimizer.py
├── config.py (配置参数)
└── re (标准库)
```

---

### 2.4 向量存储层

#### `chroma_utils.py` - ChromaDB 工具（正在使用）
**作用**：ChromaDB 向量数据库操作

**主要功能**：
- 初始化 ChromaDB 客户端
- 创建/获取 Collection
- 索引数据（生成嵌入向量并存储）
- 检索相似文档

**依赖关系**：
```
chroma_utils.py
├── chromadb (向量数据库)
├── config.py (配置参数)
└── streamlit (UI 反馈)
```

**关键代码**：
```python
# 1. 初始化客户端
def get_chroma_client():
    client = chromadb.PersistentClient(path=".")
    return client

# 2. 创建 Collection
def setup_chroma_collection(_client):
    collection = _client.get_or_create_collection(
        name="medical_rag_lite",
        metadata={"hnsw:space": "cosine"}
    )

# 3. 索引数据
def index_data_if_needed(client, data, embedding_model):
    # 生成嵌入向量
    embeddings = embedding_model.encode(docs_for_embedding)
    
    # 插入数据库
    collection.add(
        ids=ids,
        embeddings=embeddings.tolist(),
        documents=docs_for_embedding,
        metadatas=metadatas
    )

# 4. 检索文档
def search_similar_documents(client, query, embedding_model, top_k):
    query_embedding = embedding_model.encode([query])[0]
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k
    )
    return hit_ids, distances
```

---

#### `milvus_utils.py` - Milvus 工具（已废弃）
**作用**：Milvus 向量数据库操作（未使用）

**说明**：项目最初设计使用 Milvus Lite，后来改用 ChromaDB，但保留了此文件。

---

### 2.5 模型层

#### `models.py` - 模型加载
**作用**：加载嵌入模型和生成模型

**主要功能**：
- 加载嵌入模型（moka-ai/m3e-base）
- 加载生成模型（Ollama qwen3:8b）

**依赖关系**：
```
models.py
├── sentence-transformers (嵌入模型)
├── config.py (配置参数)
└── streamlit (UI 反馈)
```

**关键代码**：
```python
# 1. 加载嵌入模型
@st.cache_resource
def load_embedding_model(model_name):
    model = SentenceTransformer(model_name)
    return model

# 2. 加载生成模型（Ollama）
@st.cache_resource
def load_generation_model(model_type):
    # 不需要加载，直接使用 Ollama API
    return "ollama", None
```

---

#### `retrieval_optimizer.py` - 检索优化
**作用**：高级检索功能

**主要功能**：
- 重排序文档（多维度评分）
- 混合检索（语义+关键词）
- 查询扩展
- 结果去重
- 多跳检索
- 关键词提取

**依赖关系**：
```
retrieval_optimizer.py
├── chroma_utils.py (基础检索)
├── config.py (配置参数)
├── re (标准库)
└── collections (标准库)
```

**关键代码**：
```python
# 1. 重排序
def rerank_documents(query, documents, distances, embedding_model, top_k):
    # 多维度评分
    semantic_score = 1.0 / (1.0 + distances[i])
    keyword_score = calculate_keyword_match(query_keywords, doc_content)
    length_penalty = 1.0 / (1.0 + len(doc_content) / 2000.0)
    
    # 综合分数
    combined_score = 0.6 * semantic_score + 0.3 * keyword_score + 0.1 * length_penalty
    
    return reranked_docs, reranked_distances

# 2. 混合检索
def hybrid_search(query, chroma_client, embedding_model, top_k):
    # 向量检索
    retrieved_ids, distances = search_similar_documents(...)
    
    # 获取文档
    retrieved_docs = [id_to_doc_map[id] for id in retrieved_ids]
    
    # 重排序
    reranked_docs, reranked_distances = rerank_documents(...)
    
    return reranked_docs, reranked_distances

# 3. 查询扩展
def query_expansion(query, embedding_model, chroma_client):
    keywords = extract_keywords(query)
    expanded_queries = [query]
    
    # 生成查询变体
    expanded_queries.append(f"{keywords[0]} {keywords[1]}")
    
    return expanded_queries
```

---

### 2.6 配置层

#### `config.py` - 系统配置
**作用**：系统全局配置参数

**主要配置**：
```python
# 数据配置
DATA_FILE = "./data/processed_data.json"
COLLECTION_NAME = "medical_rag_lite"

# 模型配置
EMBEDDING_MODEL_NAME = 'moka-ai/m3e-base'
GENERATION_MODEL_NAME = "ollama"
OLLAMA_MODEL = "qwen3:8b"
OLLAMA_BASE_URL = "http://localhost:11434"
EMBEDDING_DIM = 768

# 检索参数
MAX_ARTICLES_TO_INDEX = 500
TOP_K = 5

# 生成参数
MAX_NEW_TOKENS_GEN = 1024
TEMPERATURE = 0.7
TOP_P = 0.9
REPETITION_PENALTY = 1.1

# 全局映射
id_to_doc_map = {}
```

**依赖关系**：
```
config.py
└── 无依赖（纯配置）
```

---

#### `requirements.txt` - Python 依赖包
```
streamlit
pymilvus (未使用)
sentence-transformers
transformers
torch
accelerate
```

---

### 2.7 测试与评估层

#### `test_rag_performance.py` - RAG 性能测试
**作用**：测试 RAG 系统性能

**主要功能**：
- 测试检索相关性
- 测试生成质量
- 测试响应时间
- 生成性能报告

**依赖关系**：
```
test_rag_performance.py
├── config.py
├── data_utils.py
├── models.py
├── chroma_utils.py
├── rag_core.py
└── retrieval_optimizer.py
```

---

#### `evaluate_rag.py` - RAG 评估
**作用**：评估 RAG 系统并生成报告

**主要功能**：
- 运行 6 个测试查询
- 评估检索相关性
- 评估生成质量
- 生成 `eval_generation.json`

**依赖关系**：
```
evaluate_rag.py
├── config.py
├── data_utils.py
├── models.py
├── chroma_utils.py
├── rag_core.py
└── retrieval_optimizer.py
```

---

#### `diagnostics.py` - 系统诊断
**作用**：自动化诊断系统状态

**主要功能**：
- 环境检测
- 依赖包检测
- 嵌入模型检测
- Ollama 检测
- ChromaDB 检测
- 数据检测

---

---

## 三、数据流程图

```
原始数据（sources/*.txt）
    ↓
预处理（preprocess.py）
    ↓
JSON 数据（data/processed_data.json）
    ↓
数据加载（data_utils.py）
    ↓
分块优化（chunk_optimizer.py）
    ↓
嵌入模型（models.py → m3e-base）
    ↓
向量索引（chroma_utils.py → ChromaDB）
    ↓
用户查询（app.py）
    ↓
检索优化（retrieval_optimizer.py）
    ├── 混合检索
    ├── 重排序
    ├── 查询扩展
    └── 去重
    ↓
上下文构建（rag_core.py）
    ↓
答案生成（rag_core.py → Ollama API）
    ↓
结果显示（app.py → Streamlit）
```

---

## 四、文件依赖关系图

```
app.py (主入口)
├── config.py
├── data_utils.py
│   └── json
├── models.py
│   ├── sentence-transformers
│   └── config.py
├── chroma_utils.py
│   ├── chromadb
│   └── config.py
├── rag_core.py
│   ├── config.py
│   └── requests
└── retrieval_optimizer.py
    ├── chroma_utils.py
    ├── config.py
    └── re, collections

test_rag_performance.py
├── config.py
├── data_utils.py
├── models.py
├── chroma_utils.py
├── rag_core.py
└── retrieval_optimizer.py

evaluate_rag.py
├── config.py
├── data_utils.py
├── models.py
├── chroma_utils.py
├── rag_core.py
└── retrieval_optimizer.py

preprocess.py
├── os
├── json
└── re
```

---

## 五、核心模块交互流程

### 5.1 系统启动流程

```
用户启动 app.py
    ↓
1. 加载配置 (config.py)
    ↓
2. 初始化 ChromaDB (chroma_utils.py)
    ↓
3. 加载嵌入模型 (models.py)
    ↓
4. 加载生成模型 (models.py)
    ↓
5. 加载数据 (data_utils.py)
    ↓
6. 索引数据 (chroma_utils.py)
    ↓
7. 显示 Web 界面 (Streamlit)
```

### 5.2 查询处理流程

```
用户输入查询
    ↓
1. 检索优化 (retrieval_optimizer.py)
    ├── 混合检索（语义+关键词）
    ├── 查询扩展
    ├── 重排序
    └── 去重
    ↓
2. 上下文构建 (rag_core.py)
    ├── 提取文档内容
    ├── 添加文档元数据
    └── 构建上下文字符串
    ↓
3. 提示词工程 (rag_core.py)
    ├── 设计提示词模板
    └── 填充上下文和查询
    ↓
4. 答案生成 (rag_core.py)
    ├── 调用 Ollama API
    ├── 超时处理
    └── 错误处理
    ↓
5. 结果显示 (app.py)
    ├── 显示检索文档
    ├── 显示相似度分数
    └── 显示生成答案
```

---

## 六、关键数据结构

### 6.1 文档数据结构（processed_data.json）

```json
{
    "id": "吴银根.txt_0",
    "title": "吴银根",
    "abstract": "博及医源，深耕临床...",
    "source_file": "吴银根.txt",
    "chunk_index": 0
}
```

### 6.2 id_to_doc_map 映射结构

```python
{
    0: {
        'title': '吴银根',
        'abstract': '博及医源，深耕临床...',
        'content': 'Title: 吴银根\nAbstract: 博及医源...',
        'source_file': '吴银根.txt'
    },
    1: {...},
    ...
}
```

### 6.3 检索结果结构

```python
{
    'ids': [[328, 331, 337, ...]],
    'distances': [[0.2186, 0.2204, 0.2326, ...]],
    'documents': [['Title: ...', 'Title: ...', ...]],
    'metadatas': [[
        {'title': '吴银根', 'source_file': '吴银根.txt', ...},
        ...
    ]]
}
```

---

## 七、技术栈总结

### 7.1 前端
- **Streamlit**：Web 应用框架

### 7.2 后端
- **Python**：主要编程语言
- **Ollama**：本地 LLM 服务（qwen3:8b）

### 7.3 向量数据库
- **ChromaDB**：向量数据库（正在使用）
- **Milvus Lite**：向量数据库（已废弃）

### 7.4 嵌入模型
- **moka-ai/m3e-base**：中文嵌入模型（768维）

### 7.5 工具库
- **sentence-transformers**：嵌入模型框架
- **transformers**：HuggingFace 模型库
- **torch**：深度学习框架
- **requests**：HTTP 请求库

---

## 八、关键设计模式

### 8.1 模块化设计
- 每个功能模块独立
- 通过配置文件统一管理
- 便于维护和扩展

### 8.2 缓存机制
- 使用 `@st.cache_resource` 缓存模型
- 避免重复加载

### 8.3 错误处理
- 超时处理
- 连接错误处理
- 友好的错误提示

### 8.4 配置管理
- 集中式配置（config.py）
- 便于调优参数

---

**创建时间**：2026-01-07
**版本**：v1.0