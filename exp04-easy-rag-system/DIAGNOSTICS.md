# RAG 系统检测方案

## 检测目标

确保 RAG 系统的所有组件正常工作，包括：
1. Python 环境
2. 依赖包
3. 嵌入模型
4. 生成模型 (Ollama)
5. 向量数据库 (ChromaDB)
6. 数据文件
7. 检索功能
8. 生成功能

---

## 自动化检测脚本

### 快速检测

```powershell
# 运行所有检测
python diagnostics.py
```

### 检测项目

#### 1. 环境检测

**检测内容**:
- Python 版本
- 工作目录
- 环境变量

**期望结果**:
- Python 3.10+
- 正确的工作目录
- 必要的环境变量已设置

#### 2. 依赖包检测

**检测内容**:
- streamlit
- sentence-transformers
- chromadb
- requests
- torch

**期望结果**:
- 所有包已安装
- 版本兼容

#### 3. 模型检测

**嵌入模型**:
- 模型文件存在
- 模型可以加载
- 模型可以编码文本

**生成模型**:
- Ollama 服务运行
- qwen3:8b 模型已下载
- Ollama API 可访问

#### 4. 数据库检测

**检测内容**:
- ChromaDB 文件存在
- Collection 已创建
- 数据已索引

**期望结果**:
- 数据库可访问
- Collection 包含数据

#### 5. 数据检测

**检测内容**:
- 数据文件存在
- 数据格式正确
- 数据数量充足

**期望结果**:
- 数据文件可读取
- 至少包含 100 条记录

#### 6. 功能检测

**检索功能**:
- 可以检索文档
- 检索结果相关

**生成功能**:
- 可以生成答案
- 答案格式正确

---

## 手动检测步骤

### 步骤 1: 环境检测

```powershell
# 检查 Python 版本
python --version

# 检查工作目录
pwd

# 检查环境变量
echo $env:HF_HOME
```

### 步骤 2: 依赖包检测

```powershell
# 检查已安装的包
pip list

# 验证关键包
python -c "import streamlit; print('streamlit:', streamlit.__version__)"
python -c "import sentence_transformers; print('sentence-transformers:', sentence_transformers.__version__)"
python -c "import chromadb; print('chromadb:', chromadb.__version__)"
```

### 步骤 3: 模型检测

```powershell
# 检测嵌入模型
python test_models.py

# 检测 Ollama
curl http://localhost:11434/api/tags
ollama list
```

### 步骤 4: 数据库检测

```powershell
# 检查数据库文件
Test-Path chroma.sqlite3

# 检查 Collection
python -c "import chromadb; client = chromadb.PersistentClient('.'); print(client.list_collections())"
```

### 步骤 5: 数据检测

```powershell
# 检查数据文件
Test-Path data/processed_data.json

# 检查数据内容
python -c "import json; data = json.load(open('data/processed_data.json')); print(f'Total documents: {len(data)}')"
```

### 步骤 6: 功能检测

```powershell
# 启动 Web 界面
streamlit run app.py

# 或运行性能测试
python test_rag_performance.py
```

---

## 性能基准

### 响应时间基准

| 操作 | 期望时间 | 最大可接受时间 |
|------|---------|---------------|
| 系统初始化 | < 30s | < 60s |
| 模型加载 | < 20s | < 40s |
| 数据索引 | < 60s | < 120s |
| 单次查询 | < 10s | < 20s |

### 质量基准

| 指标 | 优秀 | 良好 | 可接受 |
|------|------|------|--------|
| 检索相关性 | > 80% | > 70% | > 60% |
| 生成质量 | > 80% | > 70% | > 60% |
| 答案完整性 | > 90% | > 80% | > 70% |

---

## 定期检测计划

### 每日检测
- 系统启动检测
- 关键服务状态检测

### 每周检测
- 完整功能检测
- 性能基准测试
- 数据完整性检测

### 每月检测
- 依赖包更新检测
- 模型性能评估
- 系统优化建议

---

## 检测报告模板

```markdown
# RAG 系统检测报告

**检测日期**: 2026-01-07
**检测人员**: [姓名]
**系统版本**: v2.0

## 检测结果

### 环境检测
- [ ] Python 版本: 3.13.9
- [ ] 工作目录: 正确
- [ ] 环境变量: 已设置

### 依赖包检测
- [ ] streamlit: 已安装
- [ ] sentence-transformers: 已安装
- [ ] chromadb: 已安装
- [ ] requests: 已安装

### 模型检测
- [ ] 嵌入模型: 可加载
- [ ] 生成模型: 可访问

### 数据库检测
- [ ] ChromaDB: 可访问
- [ ] Collection: 已创建
- [ ] 数据: 已索引

### 功能检测
- [ ] 检索功能: 正常
- [ ] 生成功能: 正常

## 性能指标

- 系统初始化: XX 秒
- 模型加载: XX 秒
- 单次查询: XX 秒
- 检索相关性: XX%
- 生成质量: XX%

## 问题与建议

[记录发现的问题和改进建议]

## 结论

[总体评估]
```

---

## 检测工具

### 1. 自动化检测脚本
```powershell
python diagnostics.py
```

### 2. 模型测试脚本
```powershell
python test_models.py
```

### 3. 性能测试脚本
```powershell
python test_rag_performance.py
```

### 4. Web 界面测试
```powershell
streamlit run app.py
```

---

## 检测失败处理

### 如果检测失败

1. **查看错误日志**
   - 检查控制台输出
   - 查看错误堆栈

2. **参考排查方案**
   - 查看 TROUBLESHOOTING.md
   - 按照步骤排查

3. **重新运行检测**
   - 修复问题后重新检测
   - 确认问题已解决

4. **记录问题**
   - 记录问题详情
   - 记录解决方案

---

## 检测最佳实践

1. **定期检测**: 按照计划定期执行检测
2. **记录结果**: 保存检测报告
3. **及时修复**: 发现问题及时处理
4. **持续优化**: 根据检测结果优化系统

---

**更新日期**: 2026-01-07
**版本**: v1.0