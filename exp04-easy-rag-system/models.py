import streamlit as st
from sentence_transformers import SentenceTransformer
import requests
import os
from config import OLLAMA_BASE_URL, OLLAMA_MODEL

@st.cache_resource
def load_embedding_model(model_name):
    """Loads the sentence transformer model."""
    st.write(f"Loading embedding model: {model_name}...")

    # 强制使用本地缓存，禁用在线检查
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['HF_DATASETS_OFFLINE'] = '1'
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'

    try:
        # 方法1: 尝试直接使用本地路径加载
        cache_path = os.path.abspath('./hf_cache')
        st.write(f"Using cache folder: {cache_path}")

        # 检查本地模型路径
        local_model_path = os.path.join(
            cache_path,
            'hub',
            'models--moka-ai--m3e-base',
            'snapshots',
            '764b537a0e50e5c7d64db883f2d2e051cbe3c64c'
        )

        if os.path.exists(local_model_path):
            st.write(f"Loading model from local path: {local_model_path}")
            model = SentenceTransformer(local_model_path, device='cpu')
            st.success("Embedding model loaded from local cache.")
            return model

        # 方法2: 尝试使用 cache_folder 参数
        st.write(f"Attempting to load with cache_folder parameter...")
        model = SentenceTransformer(model_name, cache_folder=cache_path, device='cpu')
        st.success("Embedding model loaded.")
        return model

    except Exception as e:
        st.error(f"Failed to load embedding model: {e}")
        st.info("Please ensure the model is downloaded and available in the cache folder.")
        return None

@st.cache_resource
def load_generation_model(model_name):
    """Loads the Ollama API client."""
    st.write(f"Using Ollama API for generation: {OLLAMA_MODEL}...")
    try:
        # 测试 ollama 连接
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags")
        if response.status_code == 200:
            st.success(f"Ollama API connected. Using model: {OLLAMA_MODEL}")
            return OLLAMA_MODEL, None  # 返回模型名称，不需要 tokenizer
        else:
            st.error(f"Failed to connect to Ollama API at {OLLAMA_BASE_URL}")
            return None, None
    except Exception as e:
        st.error(f"Failed to connect to Ollama API: {e}")
        return None, None 