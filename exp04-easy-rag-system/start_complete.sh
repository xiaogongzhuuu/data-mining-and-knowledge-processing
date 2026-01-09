#!/bin/bash
# 医疗 RAG 系统完整启动脚本
# 包括清理、验证和启动应用的所有步骤

set -e  # 如果任何命令失败就退出

echo "=================================================="
echo "🚀 医疗 RAG 系统启动脚本"
echo "=================================================="

# 步骤1：检查环境
echo ""
echo "[步骤 1] 检查 Python 环境..."
python3 --version
if ! command -v pip3 &> /dev/null; then
    echo "❌ 找不到 pip3，请确保已安装 Python3"
    exit 1
fi
echo "✅ Python 环境正常"

# 步骤2：检查依赖
echo ""
echo "[步骤 2] 检查必要的 Python 包..."
python3 -c "import chromadb; print(f'✅ chromadb {chromadb.__version__}')" || {
    echo "❌ chromadb 未安装，请运行: pip3 install chromadb"
    exit 1
}
python3 -c "import sentence_transformers; print('✅ sentence-transformers')" || {
    echo "❌ sentence-transformers 未安装，请运行: pip3 install sentence-transformers"
    exit 1
}
python3 -c "import streamlit; print('✅ streamlit')" || {
    echo "❌ streamlit 未安装，请运行: pip3 install streamlit"
    exit 1
}

# 步骤3：检查缓存模型
echo ""
echo "[步骤 3] 检查嵌入模型缓存..."
if [ -d "./hf_cache/models--moka-ai--m3e-base" ]; then
    echo "✅ 嵌入模型缓存存在"
else
    echo "⚠️  嵌入模型缓存不存在，应用会尝试加载..."
fi

# 步骤4：清理旧数据（可选）
echo ""
echo "[步骤 4] 清理旧的 ChromaDB 数据..."
if [ -d "./chroma_data" ]; then
    rm -rf ./chroma_data
    echo "✅ 已删除旧数据"
else
    echo "✅ 没有旧数据需要清理"
fi

# 步骤5：运行离线测试（可选）
echo ""
echo "[步骤 5] 运行离线模式测试..."
read -p "是否运行测试? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    python3 test_chroma_offline.py
    if [ $? -eq 0 ]; then
        echo "✅ 离线测试通过"
    else
        echo "❌ 离线测试失败"
        exit 1
    fi
else
    echo "⏭️  跳过测试"
fi

# 步骤6：启动应用
echo ""
echo "=================================================="
echo "✅ 所有检查完成，正在启动应用..."
echo "=================================================="
echo ""
echo "💡 应用将在以下地址运行:"
echo "   http://localhost:8501"
echo ""
echo "📝 如需启用生成功能，请在另一个终端运行:"
echo "   ollama serve"
echo ""
echo "按 Ctrl+C 停止应用"
echo ""

streamlit run app.py
