# Milvus Lite Configuration
MILVUS_LITE_DATA_PATH = "./milvus_lite_data.db" # Path to store Milvus Lite data
COLLECTION_NAME = "medical_rag_lite" # Use a different name if needed

# Data Configuration
DATA_FILE = "./data/processed_data.json"

# Model Configuration
# 使用更轻量的中文嵌入模型
EMBEDDING_MODEL_NAME = 'moka-ai/m3e-base'
GENERATION_MODEL_NAME = "ollama"  # 使用 ollama API
OLLAMA_MODEL = "qwen3:8b"  # ollama 部署的模型
OLLAMA_BASE_URL = "http://localhost:11434"  # ollama API 地址
EMBEDDING_DIM = 768 # m3e-base 的维度

# Indexing and Search Parameters
MAX_ARTICLES_TO_INDEX = 500
TOP_K = 5  # 增加到 5 以提高召回率
# Milvus index parameters (adjust based on data size and needs)
INDEX_METRIC_TYPE = "L2" # Or "IP"
INDEX_TYPE = "IVF_FLAT"  # Milvus Lite 支持的索引类型
# HNSW index params (adjust as needed)
INDEX_PARAMS = {"nlist": 128}
# HNSW search params (adjust as needed)
SEARCH_PARAMS = {"nprobe": 16}

# Generation Parameters
MAX_NEW_TOKENS_GEN = 1024  # 增加到1024，避免回答截断
TEMPERATURE = 0.7
TOP_P = 0.9
REPETITION_PENALTY = 1.1

# Global map to store document content (populated during indexing)
# Key: document ID (int), Value: dict {'title': str, 'abstract': str, 'content': str}
id_to_doc_map = {} 