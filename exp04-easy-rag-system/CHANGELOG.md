# 项目变更日志 (CHANGELOG)

## 2026-01-07 - 问题排查和诊断系统

### 新增功能
- **自动化诊断脚本** (`diagnostics.py`)
  - 环境检测（Python 版本、工作目录、环境变量）
  - 依赖包检测（streamlit、sentence-transformers、chromadb 等）
  - 嵌入模型检测（模型文件、加载测试、编码测试）
  - Ollama 检测（服务状态、模型可用性）
  - ChromaDB 检测（数据库文件、Collection 状态）
  - 数据检测（数据文件、数据完整性）
  - 自动生成检测报告

- **模型测试脚本** (`test_models.py`)
  - 模型缓存检查
  - 模型文件完整性验证
  - 模型加载测试
  - 编码功能测试

### 修改内容
- **models.py**
  - 修复嵌入模型加载问题
  - 添加本地模型路径直接加载
  - 改进错误处理和日志输出
  - 强制使用离线模式

### 新增文档
- **TROUBLESHOOTING.md** - 问题排查方案
  - 完整的问题诊断流程
  - 常见问题解决方案
  - 错误代码对照表
  - 紧急恢复方案

- **DIAGNOSTICS.md** - 检测方案
  - 检测目标和基准
  - 自动化检测工具
  - 性能基准指标
  - 定期检测计划
  - 检测报告模板

### 修复问题
- **嵌入模型加载失败**
  - 问题：SentenceTransformer 尝试从 HuggingFace 下载模型
  - 原因：本地缓存路径配置不正确
  - 解决方案：直接使用本地模型路径加载，强制离线模式

- **Ollama 连接问题**
  - 问题：Ollama 服务未启动导致连接失败
  - 解决方案：提供详细的启动和验证步骤

### 测试结果
运行自动化诊断脚本：
```
[OK] Environment
[OK] Dependencies
[OK] Embedding Model
[OK] Ollama
[OK] ChromaDB
[OK] Data

Total: 6/6 checks passed
```

所有检测项目均通过，系统已就绪。

### 使用说明
1. 运行诊断：`python diagnostics.py`
2. 测试模型：`python test_models.py`
3. 查看排查方案：`TROUBLESHOOTING.md`
4. 查看检测方案：`DIAGNOSTICS.md`

---

## 2026-01-07 - RAG 系统性能优化

### 新增功能
- **高级检索模块** (`retrieval_optimizer.py`)
  - 实现重排序机制（rerank_documents）：结合语义相似度和关键词匹配
  - 实现混合检索（hybrid_search）：向量检索 + 关键词检索
  - 实现查询扩展（query_expansion）：生成查询变体提升召回率
  - 实现结果去重（remove_duplicate_documents）：避免重复内容
  - 实现多跳检索（multi_hop_retrieval）：迭代检索获取更全面信息
  - 关键词提取（extract_keywords）：智能提取查询关键词

- **数据分块优化模块** (`chunk_optimizer.py`)
  - 智能文本分块（smart_chunk_text）：在语义边界处切分
  - 优化数据分块策略（optimize_data_chunks）：支持重叠分块
  - 增强文档元数据（enhance_document_metadata）：添加检索线索
  - 文档关键词提取（extract_document_keywords）：提取中医专业术语

### 修改内容
- **rag_core.py**
  - 优化提示词工程：添加详细的任务要求和回答格式
  - 增加上下文长度：从 800 字符增加到 1200 字符
  - 添加文档标题和来源信息到上下文
  - 改进错误处理：区分超时、连接错误等不同错误类型
  - 要求生成答案时包含参考来源

- **app.py**
  - 集成混合检索功能：使用 hybrid_search 替代简单检索
  - 添加结果去重：调用 remove_duplicate_documents
  - 改进检索结果显示：显示相似度分数和来源信息
  - 优化用户界面：更清晰的文档展示

- **test_rag_performance.py**
  - 扩展测试查询集：从 4 个增加到 6 个测试用例
  - 集成优化功能：使用混合检索和重排序
  - 更新性能评级标准：更细致的评级（优秀/良好/中等/需要改进）
  - 添加优化建议：根据测试结果提供针对性建议

### 优化效果
- **检索相关性提升**：
  - 重排序机制提高检索准确性
  - 混合检索结合语义和关键词匹配
  - 查询扩展提升召回率

- **生成质量提升**：
  - 改进的提示词工程提供更明确的指导
  - 增加上下文长度保留更多信息
  - 要求包含参考来源提升可信度

- **架构优化**：
  - 模块化设计，易于扩展和维护
  - 支持多种检索策略组合
  - 提供灵活的配置选项

### 技术亮点
- **多维度检索评估**：语义相似度 + 关键词匹配 + 文档长度
- **智能分块策略**：在句子、段落边界切分，支持重叠
- **中医领域优化**：提取中医专业术语作为关键词
- **错误处理增强**：区分不同错误类型，提供友好提示

### 待验证
- [ ] 运行性能测试验证优化效果
- [ ] 根据测试结果进一步调优参数
- [ ] 评估优化后的响应时间
- [ ] 验证多跳检索的实际效果

### 配置参数调整建议
- **TOP_K**: 可根据需要从 3 增加到 5-10 以提升召回率
- **chunk_size**: 可根据文档长度调整（建议 800-1500）
- **overlap**: 建议设置为 chunk_size 的 10-20%
- **重排序权重**: 可根据实际效果调整（当前：语义 0.6，关键词 0.3，长度 0.1）

---

## 2026-01-06 - 性能优化和问题修复

### 修复问题
- **ChromaDB 初始化卡住问题**
  - 问题：ChromaDB 客户端初始化时卡住，无法正常启动
  - 原因：sentence-transformers 模型加载超时，all-MiniLM-L6-v2 模型加载时间过长
  - 解决方案：更换为更轻量的中文嵌入模型 moka-ai/m3e-base，并优化模型加载配置

### 修改内容
- **config.py**:
  - EMBEDDING_MODEL_NAME: 'all-MiniLM-L6-v2' → 'moka-ai/m3e-base'
  - EMBEDDING_DIM: 384 → 768 (匹配 m3e-base 的维度)
  
- **models.py**:
  - 添加环境变量配置，强制使用本地缓存
  - 优化模型加载流程，添加 cache_folder 参数

- **chroma_utils.py**:
  - 移除 @st.cache_resource 装饰器，避免缓存冲突
  - 简化 ChromaDB 客户端初始化逻辑

### 技术栈更新
- **嵌入模型**: moka-ai/m3e-base (768维，中文优化)
- **向量数据库**: ChromaDB (持久化存储)
- **生成模型**: Ollama qwen3:8b
- **前端框架**: Streamlit

### 待验证
- [ ] 系统是否能正常启动
- [ ] 检索性能测试
- [ ] 生成质量评估

---

## 2026-01-06 - 项目初始化审查

### 当前项目状态
- **项目名称：** 医疗 RAG 系统
- **技术栈：**
  - 前端：Streamlit Web 应用
  - 向量数据库：ChromaDB
  - 嵌入模型：all-MiniLM-L6-v2 (384维)
  - 生成模型：Ollama qwen3:8b
  - 编程语言：Python

### 核心文件结构
```
exp04-easy-rag-system/
├── app.py              # Streamlit 主应用
├── config.py           # 配置文件
├── chroma_utils.py     # ChromaDB 工具函数
├── rag_core.py         # RAG 核心逻辑
├── models.py           # 模型加载函数
├── data_utils.py       # 数据处理工具
├── requirements.txt    # 依赖包
├── data/               # 数据目录
│   └── processed_data.json  # 已处理的医疗文献数据
└── sources/            # 原始数据
    ├── 邱佳信.txt
    ├── 邵长荣.txt
    ├── 施杞.txt
    ├── 唐汉钧.txt
    ├── 王大增.txt
    ├── 王育群.txt
    ├── 吴银根.txt
    ├── 徐蓉娟.txt
    └── 徐振晔.txt
```

### 功能模块
1. **数据加载** (`data_utils.py`)
   - 从 JSON 文件加载预处理数据

2. **模型加载** (`models.py`)
   - 嵌入模型：SentenceTransformer
   - 生成模型：Ollama API 客户端

3. **向量存储** (`chroma_utils.py`)
   - ChromaDB 客户端初始化
   - Collection 创建和配置
   - 数据索引（嵌入生成 + 插入）
   - 相似文档检索

4. **RAG 核心** (`rag_core.py`)
   - 上下文构建
   - 通过 Ollama API 生成答案

5. **Web UI** (`app.py`)
   - Streamlit 界面
   - 查询输入
   - 结果展示（检索文档 + 生成答案）

### 已知问题/待确认项
- [ ] Ollama 服务是否已启动 (localhost:11434)
- [ ] qwen3:8b 模型是否已下载
- [ ] ChromaDB 数据是否已索引
- [ ] 数据源文件是否完整

### 配置参数
- **向量存储路径：** `./milvus_lite_data.db` (实际使用 ChromaDB)
- **Collection 名称：** `medical_rag_lite`
- **嵌入维度：** 384
- **最大索引文档数：** 500
- **检索 Top-K：** 3
- **生成参数：**
  - 最大新令牌：512
  - 温度：0.7
  - Top-P：0.9
  - 重复惩罚：1.1

---

## 变更记录格式说明

每次修改项目时，请在此文件顶部添加新的条目，格式如下：

```markdown
## YYYY-MM-DD - 简短描述

### 新增功能
- 描述新增的功能

### 修改内容
- 描述修改的代码或配置

### 修复问题
- 描述修复的 bug

### 文档更新
- 描述文档的更新

### 备注
- 其他需要记录的信息
```