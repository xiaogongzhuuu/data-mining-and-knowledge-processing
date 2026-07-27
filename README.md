# 数据挖掘与知识处理

数据挖掘与知识处理课程实验，包含数据预处理、情感分类、知识图谱构建和 RAG 检索系统。

## 实验列表

### exp01 - 数据预处理
- 文本数据清洗与特征提取
- Word2Vec 词向量训练与降维可视化（t-SNE）
- 词云生成与情感分析

### exp02 - 情感分类
三类模型对比实验：

| 模型 | 说明 |
|------|------|
| TextCNN | 卷积神经网络文本分类 |
| BERT | 预训练语言模型微调 |
| Qwen | 大语言模型情感分类 |

每个模型包含完整的训练、评估、对比分析脚本。

### exp03 - 医学知识图谱
- 基于大模型的实体关系抽取
- Neo4j 图数据库存储与查询
- 医学领域知识图谱构建

### exp04 - RAG 检索系统
- 基于 ChromaDB + M3E 嵌入的文档检索
- 中医名医经验知识库问答
- Streamlit Web 交互界面
- 检索优化与离线测试

## 技术栈

- **深度学习**: PyTorch, Transformers, BERT, Qwen
- **NLP**: Word2Vec, TextCNN
- **图数据库**: Neo4j
- **RAG**: LangChain, ChromaDB, M3E Embedding
- **Web**: Streamlit, FastAPI
