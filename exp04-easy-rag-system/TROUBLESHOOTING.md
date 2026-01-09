# RAG 系统问题排查方案

## 问题诊断流程

### 1. 环境检查

#### 1.1 Python 环境
```powershell
python --version
# 期望输出: Python 3.13.9
```

#### 1.2 依赖包检查
```powershell
pip list | grep -E "streamlit|sentence-transformers|chromadb|requests"
```

#### 1.3 目录结构检查
```powershell
cd D:\data-mining-and-knowledge-processing\2025-spring\exp04-easy-rag-system
dir
```

期望看到以下关键文件：
- app.py
- config.py
- models.py
- chroma_utils.py
- rag_core.py
- data/processed_data.json
- hf_cache/

---

### 2. 模型加载问题

#### 症状
```
Failed to load embedding model: We couldn't connect to 'https://huggingface.co'
```

#### 原因
- SentenceTransformer 尝试从 HuggingFace 下载模型
- 本地缓存路径配置不正确

#### 解决方案

**步骤 1**: 检查模型缓存是否存在
```powershell
Test-Path "D:\data-mining-and-knowledge-processing\2025-spring\exp04-easy-rag-system\hf_cache\hub\models--moka-ai--m3e-base"
# 期望输出: True
```

**步骤 2**: 检查模型文件完整性
```powershell
Get-ChildItem "D:\data-mining-and-knowledge-processing\2025-spring\exp04-easy-rag-system\hf_cache\hub\models--moka-ai--m3e-base\snapshots\764b537a0e50e5c7d64db883f2d2e051cbe3c64c"
# 期望看到: model.safetensors (约 390 MB)
```

**步骤 3**: 运行模型测试
```powershell
python test_models.py
# 期望输出: [SUCCESS] All tests passed!
```

**步骤 4**: 如果测试失败，重新下载模型
```powershell
# 临时启用在线模式
$env:HF_HUB_OFFLINE = "0"
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('moka-ai/m3e-base', cache_folder='./hf_cache')"
# 重新禁用在线模式
$env:HF_HUB_OFFLINE = "1"
```

---

### 3. Ollama 连接问题

#### 症状
```
Failed to connect to Ollama API: HTTPConnectionPool(host='localhost', port=11434): Max retries exceeded
```

#### 原因
- Ollama 服务未启动
- Ollama 服务端口被占用

#### 解决方案

**步骤 1**: 检查 Ollama 是否安装
```powershell
Test-Path "C:\Users\king\AppData\Local\Programs\Ollama\ollama.exe"
# 期望输出: True
```

**步骤 2**: 启动 Ollama 服务
```powershell
ollama serve
# 保持此窗口运行
```

**步骤 3**: 验证 Ollama 服务
```powershell
# 在新的 PowerShell 窗口中
curl http://localhost:11434/api/tags
# 期望输出: JSON 格式的模型列表
```

**步骤 4**: 检查所需模型
```powershell
ollama list
# 期望看到: qwen3:8b
```

**步骤 5**: 如果模型不存在，下载模型
```powershell
ollama pull qwen3:8b
```

---

### 4. ChromaDB 问题

#### 症状
```
Failed to initialize ChromaDB client
```

#### 原因
- ChromaDB 数据库文件损坏
- 权限问题

#### 解决方案

**步骤 1**: 检查 ChromaDB 文件
```powershell
Test-Path "D:\data-mining-and-knowledge-processing\2025-spring\exp04-easy-rag-system\chroma.sqlite3"
```

**步骤 2**: 如果数据库损坏，重建
```powershell
# 备份旧数据库
Move-Item chroma.sqlite3 chroma.sqlite3.bak -ErrorAction SilentlyContinue
# 重新索引数据（运行 app.py 时会自动创建）
```

---

### 5. 数据加载问题

#### 症状
```
Unable to load data from ./data/processed_data.json
```

#### 原因
- 数据文件不存在
- 数据文件格式错误

#### 解决方案

**步骤 1**: 检查数据文件
```powershell
Test-Path "D:\data-mining-and-knowledge-processing\2025-spring\exp04-easy-rag-system\data\processed_data.json"
```

**步骤 2**: 验证数据格式
```powershell
python -c "import json; data = json.load(open('./data/processed_data.json')); print(f'Documents: {len(data)}')"
# 期望输出: Documents: [数字]
```

**步骤 3**: 如果数据损坏，重新处理
```powershell
python preprocess.py
```

---

## 常见错误代码

| 错误信息 | 原因 | 解决方案 |
|---------|------|---------|
| `Connection refused` | Ollama 服务未启动 | 运行 `ollama serve` |
| `Model not found` | 模型未下载 | 运行 `ollama pull qwen3:8b` |
| `File not found` | 路径错误 | 检查工作目录 |
| `Permission denied` | 权限问题 | 以管理员身份运行 |
| `Out of memory` | 内存不足 | 减少并发或使用更小的模型 |

---

## 日志检查

### Streamlit 日志
Streamlit 会在控制台输出详细日志，包括：
- 模型加载状态
- 数据库连接状态
- 查询处理过程
- 错误堆栈信息

### 关键日志关键词
- `[OK]` - 操作成功
- `[FAILED]` - 操作失败
- `Error` - 错误信息
- `Warning` - 警告信息

---

## 性能优化建议

### 1. 模型加载优化
- 使用 `@st.cache_resource` 缓存模型
- 首次加载后，模型会保存在内存中

### 2. 检索优化
- 调整 `TOP_K` 参数（默认 3）
- 使用重排序机制提高准确性

### 3. 生成优化
- 调整 `MAX_NEW_TOKENS_GEN`（默认 512）
- 调整 `TEMPERATURE`（默认 0.7）

---

## 紧急恢复方案

如果系统完全无法运行：

### 方案 1: 重置配置
```powershell
# 备份当前配置
Copy-Item config.py config.py.bak
# 使用默认配置
# （手动编辑 config.py，恢复默认值）
```

### 方案 2: 重新安装依赖
```powershell
pip uninstall -y streamlit sentence-transformers chromadb requests
pip install -r requirements.txt
```

### 方案 3: 清除缓存
```powershell
# 备份数据
Copy-Item chroma.sqlite3 chroma.sqlite3.bak
# 清除缓存
Remove-Item -Recurse -Force .streamlit
Remove-Item -Recurse -Force __pycache__
```

---

## 联系支持

如果问题仍未解决：
1. 收集完整的错误日志
2. 记录复现步骤
3. 提供系统环境信息（Python 版本、操作系统等）

---

**更新日期**: 2026-01-07
**版本**: v1.0