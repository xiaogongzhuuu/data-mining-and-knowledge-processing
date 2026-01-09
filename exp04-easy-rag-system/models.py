import streamlit as st
from sentence_transformers import SentenceTransformer
import requests
import os
from config import OLLAMA_BASE_URL, OLLAMA_MODEL

@st.cache_resource
def load_embedding_model(model_name):
    """加载句子嵌入模型。"""
    # 强制离线设置
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['HF_HUB_OFFLINE'] = '1'

    try:
        cache_path = os.path.abspath('./hf_cache')
        # 将调试信息输出到终端而非页面
        print(f"DEBUG: Loading embedding model: {model_name}")
        print(f"DEBUG: Using cache folder: {cache_path}")

        # 检查本地模型快照路径（根据你的实际路径调整）
        local_model_path = os.path.join(
            cache_path,
            'hub',
            'models--moka-ai--m3e-base',
            'snapshots',
            '764b537a0e50e5c7d64db883f2d2e051cbe3c64c'
        )

        if os.path.exists(local_model_path):
            model = SentenceTransformer(local_model_path, device='cpu')
            print("DEBUG: Embedding model loaded from local cache.")
            return model
        else:
            # 回退到普通加载（依赖环境变量）
            model = SentenceTransformer(model_name, cache_folder=cache_path, device='cpu')
            print("DEBUG: Embedding model loaded.")
            return model
    except Exception as e:
        print(f"ERROR: Failed to load embedding model: {e}")
        return None

@st.cache_resource
def load_generation_model(model_name):
    """加载 Ollama 客户端。仅在终端打印状态。"""
    print(f"DEBUG: Checking Ollama API: {OLLAMA_MODEL}...")
    try:
        # 测试 ollama 连接
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        if response.status_code == 200:
            print(f"DEBUG: Ollama API connected successfully.")
            return OLLAMA_MODEL, None
        else:
            print(f"DEBUG: Failed to connect to Ollama API (HTTP {response.status_code})")
            return None, None
    except Exception as e:
        print(f"DEBUG: Ollama service not reachable: {e}")
        return None, None