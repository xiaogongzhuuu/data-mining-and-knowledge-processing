# RAG 系统启动检查清单

## 启动前检查

### 1. 环境检查
- [ ] Python 3.10+ 已安装
- [ ] 工作目录正确：`D:\data-mining-and-knowledge-processing\2025-spring\exp04-easy-rag-system`
- [ ] PowerShell 配置文件已修复（无 Conda 卡顿问题）

### 2. 依赖检查
- [ ] 运行 `pip list` 确认所有依赖已安装
- [ ] 关键包：streamlit, sentence-transformers, chromadb, requests, torch

### 3. 模型检查
- [ ] 嵌入模型缓存存在：`./hf_cache/hub/models--moka-ai--m3e-base`
- [ ] 模型文件完整：`model.safetensors` (约 390 MB)
- [ ] 运行 `python test_models.py` 验证模型加载

### 4. Ollama 检查
- [ ] Ollama 已安装：`C:\Users\king\AppData\Local\Programs\Ollama\ollama.exe`
- [ ] Ollama 服务已启动：`ollama serve`
- [ ] qwen3:8b 模型已下载：`ollama list`
- [ ] Ollama API 可访问：`curl http://localhost:11434/api/tags`

### 5. 数据检查
- [ ] 数据文件存在：`./data/processed_data.json`
- [ ] 数据可读取：至少包含 100 条记录

### 6. 数据库检查
- [ ] ChromaDB 文件存在：`chroma.sqlite3`
- [ ] Collection 已创建：`medical_rag_lite`
- [ ] 数据已索引：至少包含 100 个实体

---

## 自动化检测

运行完整诊断：

```powershell
python diagnostics.py
```

期望输出：
```
[OK] Environment
[OK] Dependencies
[OK] Embedding Model
[OK] Ollama
[OK] ChromaDB
[OK] Data

Total: 6/6 checks passed
[SUCCESS] All checks passed! System is ready to use.
```

---

## 启动步骤

### 方式 1: Web 界面

**窗口 1（Ollama 服务）**：
```powershell
ollama serve
```

**窗口 2（RAG 系统）**：
```powershell
cd D:\data-mining-and-knowledge-processing\2025-spring\exp04-easy-rag-system
streamlit run app.py
```

浏览器会自动打开：`http://localhost:8501`

### 方式 2: 性能测试

```powershell
cd D:\data-mining-and-knowledge-processing\2025-spring\exp04-easy-rag-system
python test_rag_performance.py
```

---

## 启动后验证

### 1. Web 界面验证
- [ ] 页面正常加载
- [ ] 显示系统配置信息
- [ ] ChromaDB 初始化成功
- [ ] 嵌入模型加载成功
- [ ] Ollama 连接成功

### 2. 功能验证
- [ ] 输入问题能检索到相关文档
- [ ] 能生成答案
- [ ] 答案包含参考来源
- [ ] 响应时间可接受（< 15秒）

### 3. 性能验证
运行性能测试：
```powershell
python test_rag_performance.py
```

期望结果：
- 检索相关性 > 70%
- 生成质量 > 70%
- 平均响应时间 < 15秒

---

## 常见启动问题

### 问题 1: PowerShell 启动卡顿
**症状**：PowerShell 卡在启动界面
**解决**：配置文件已修复，关闭所有 PowerShell 窗口，重新打开

### 问题 2: 嵌入模型加载失败
**症状**：`Failed to load embedding model`
**解决**：
```powershell
python test_models.py
# 如果失败，检查模型缓存路径
```

### 问题 3: Ollama 连接失败
**症状**：`Failed to connect to Ollama API`
**解决**：
```powershell
# 窗口 1
ollama serve

# 窗口 2
curl http://localhost:11434/api/tags
```

### 问题 4: 数据库错误
**症状**：`Failed to initialize ChromaDB`
**解决**：
```powershell
# 备份旧数据库
Move-Item chroma.sqlite3 chroma.sqlite3.bak
# 重新运行系统，会自动创建新数据库
```

---

## 日志检查

### Streamlit 日志
启动后，控制台会显示详细日志：

```
Initializing ChromaDB client...
ChromaDB client initialized!
Found existing collection: 'medical_rag_lite'
Collection 'medical_rag_lite' ready. Current entity count: 355
Loading embedding model: moka-ai/m3e-base...
Using cache folder: D:\...\hf_cache
[OK] Embedding model loaded from local cache.
Using Ollama API for generation: qwen3:8b...
Ollama API connected. Using model: qwen3:8b
```

### 关键日志关键词
- `[OK]` - 操作成功
- `[FAILED]` - 操作失败
- `[WARNING]` - 警告信息
- `Error` - 错误信息

---

## 性能监控

### 关键指标
- 系统初始化时间：< 30秒
- 模型加载时间：< 20秒
- 单次查询时间：< 15秒
- 检索相关性：> 70%
- 生成质量：> 70%

### 监控工具
```powershell
# 运行性能测试
python test_rag_performance.py

# 运行诊断
python diagnostics.py
```

---

## 关闭系统

### 正常关闭
1. 关闭 Web 浏览器
2. 在 PowerShell 中按 `Ctrl+C` 停止 Streamlit
3. 关闭 Ollama 服务窗口

### 清理（可选）
```powershell
# 清理缓存
Remove-Item -Recurse -Force .streamlit
Remove-Item -Recurse -Force __pycache__
```

---

## 故障恢复

### 如果启动失败

1. **查看日志**
   - 检查控制台输出
   - 记录错误信息

2. **运行诊断**
   ```powershell
   python diagnostics.py
   ```

3. **参考排查方案**
   - 查看 `TROUBLESHOOTING.md`
   - 按照步骤排查

4. **重新启动**
   - 关闭所有窗口
   - 重新执行启动步骤

---

## 联系支持

如果问题仍未解决：
1. 收集完整的错误日志
2. 记录复现步骤
3. 运行诊断并保存结果
4. 查看 `TROUBLESHOOTING.md` 和 `DIAGNOSTICS.md`

---

**更新日期**: 2026-01-07
**版本**: v1.0