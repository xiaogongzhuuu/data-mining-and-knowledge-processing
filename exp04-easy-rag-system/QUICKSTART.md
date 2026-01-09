# RAG 系统快速启动指南

## 系统要求

- Python 3.8+
- Ollama 服务（已安装 qwen3:8b 模型）
- 至少 8GB RAM
- 5GB 可用磁盘空间

## 安装步骤

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 启动 Ollama 服务

确保 Ollama 服务正在运行：

```bash
# 检查 Ollama 服务状态
curl http://localhost:11434/api/tags

# 如果服务未运行，启动 Ollama
ollama serve
```

### 3. 验证模型

确保 qwen3:8b 模型已下载：

```bash
ollama list
```

如果没有看到 qwen3:8b，下载它：

```bash
ollama pull qwen3:8b
```

## 运行系统

### 方式一：Web 界面（推荐）

```bash
streamlit run app.py
```

浏览器会自动打开 `http://localhost:8501`

### 方式二：性能测试

```bash
python test_rag_performance.py
```

## 配置说明

主要配置文件：`config.py`

```python
# 数据配置
DATA_FILE = "./data/processed_data.json"
MAX_ARTICLES_TO_INDEX = 500

# 模型配置
EMBEDDING_MODEL_NAME = 'moka-ai/m3e-base'
OLLAMA_MODEL = "qwen3:8b"
OLLAMA_BASE_URL = "http://localhost:11434"

# 检索参数
TOP_K = 3  # 检索文档数量

# 生成参数
MAX_NEW_TOKENS_GEN = 512
TEMPERATURE = 0.7
TOP_P = 0.9
```

## 优化功能

系统已集成以下优化功能：

### 1. 混合检索
- 向量检索 + 关键词检索
- 自动重排序
- 结果去重

### 2. 智能分块
- 语义边界切分
- 重叠分块
- 关键词提取

### 3. 改进生成
- 详细提示词工程
- 增加上下文长度
- 包含参考来源

## 使用建议

### 提问技巧

1. **明确问题**：使用清晰、具体的问题
   - ✅ "吴银根的学术思想是什么？"
   - ❌ "吴银根"

2. **使用专业术语**：使用医学术语可以获得更准确的结果
   - ✅ "如何调理气血？"
   - ❌ "怎么让身体好？"

3. **多角度提问**：如果第一次回答不够详细，可以从不同角度提问
   - "肺肾两脏的关系是什么？"
   - "为什么说肺为气之主，肾为气之根？"

### 性能调优

如果需要调整性能：

1. **提高召回率**：增加 TOP_K
   ```python
   TOP_K = 5  # 从 3 增加到 5
   ```

2. **提高准确性**：调整重排序权重
   ```python
   # 在 retrieval_optimizer.py 中
   combined_score = 0.7 * semantic_score + 0.2 * keyword_score + 0.1 * length_penalty
   ```

3. **加快响应**：减少上下文长度
   ```python
   # 在 rag_core.py 中
   content = doc.get('content', '')[:800]  # 从 1200 减少到 800
   ```

## 常见问题

### Q1: 系统启动失败

**检查项**：
1. Python 版本是否 ≥ 3.8
2. 依赖包是否安装完整
3. Ollama 服务是否运行
4. 数据文件是否存在

### Q2: 检索结果不准确

**解决方案**：
1. 增加 TOP_K 参数
2. 检查数据是否正确索引
3. 尝试使用更具体的问题
4. 考虑更换嵌入模型

### Q3: 生成答案质量不佳

**解决方案**：
1. 检查检索到的文档是否相关
2. 增加上下文长度
3. 调整生成参数（temperature, top_p）
4. 优化提示词工程

### Q4: 响应速度慢

**解决方案**：
1. 减少检索文档数量（TOP_K）
2. 减少上下文长度
3. 使用更轻量的嵌入模型
4. 启用模型缓存

### Q5: Ollama 连接失败

**检查项**：
1. Ollama 服务是否运行：`curl http://localhost:11434/api/tags`
2. 模型是否已下载：`ollama list`
3. 防火墙是否阻止连接

## 性能测试

运行完整的性能测试：

```bash
python test_rag_performance.py
```

测试报告会显示：
- 检索相关性评分
- 生成质量评分
- 响应时间
- 性能评级
- 优化建议

## 文件结构

```
exp04-easy-rag-system/
├── app.py                      # Streamlit 主应用
├── config.py                   # 配置文件
├── chroma_utils.py             # ChromaDB 工具
├── rag_core.py                 # RAG 核心逻辑
├── models.py                   # 模型加载
├── data_utils.py               # 数据处理
├── retrieval_optimizer.py      # 检索优化（新增）
├── chunk_optimizer.py          # 分块优化（新增）
├── test_rag_performance.py     # 性能测试
├── requirements.txt            # 依赖包
├── CHANGELOG.md                # 变更日志
├── OPTIMIZATION_SUMMARY.md     # 优化总结
├── data/
│   └── processed_data.json     # 处理后的数据
└── sources/                    # 原始数据文件
```

## 更新日志

查看最新的更新和优化：

```bash
cat CHANGELOG.md
```

## 技术支持

如遇到问题，请检查：
1. 错误日志（控制台输出）
2. 配置文件（config.py）
3. 变更日志（CHANGELOG.md）
4. 优化总结（OPTIMIZATION_SUMMARY.md）

## 下一步

1. 运行系统并测试基本功能
2. 运行性能测试了解系统表现
3. 根据需要调整配置参数
4. 查看优化总结了解改进细节
5. 持续监控系统性能

---

**版本**：v2.0
**更新日期**：2026-01-07
**文档**：快速启动指南