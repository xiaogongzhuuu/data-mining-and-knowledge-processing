import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
import os
import sys

# 从 config 导入配置
from config import (
    MILVUS_LITE_DATA_PATH, COLLECTION_NAME, EMBEDDING_DIM,
    MAX_ARTICLES_TO_INDEX, TOP_K, id_to_doc_map, EMBEDDING_MODEL_NAME
)

class DummyEmbeddingFunction(embedding_functions.EmbeddingFunction):
    """
    虚拟嵌入函数 - 阻止 ChromaDB 尝试自动从网络下载默认模型。
    """
    def __call__(self, input):
        if isinstance(input, str):
            input = [input]
        # 返回全零向量，实际计算由 embedding_model 完成
        return [[0.0] * EMBEDDING_DIM for _ in input]

def get_chroma_client():
    """初始化并返回 ChromaDB 客户端。"""
    try:
        persist_dir = "./chroma_data"
        if not os.path.exists(persist_dir):
            os.makedirs(persist_dir, exist_ok=True)
        
        # 兼容不同版本的 ChromaDB
        client = None
        if hasattr(chromadb, "PersistentClient"):
            client = chromadb.PersistentClient(path=persist_dir)
        else:
            try:
                settings = chromadb.config.Settings(persist_directory=persist_dir)
                client = chromadb.Client(settings)
            except Exception:
                # 最后的兜底尝试
                client = chromadb.Client()
        
        print(f"DEBUG: ChromaDB client initialized at {persist_dir}")
        return client
    except Exception as e:
        print(f"ERROR: Failed to initialize ChromaDB client: {e}")
        return None

def setup_chroma_collection(_client):
    """确保 Collection 存在。成功返回 True，失败返回 False。"""
    if not _client or isinstance(_client, bool):
        return False
        
    try:
        collection_name = COLLECTION_NAME
        dummy_embedding_fn = DummyEmbeddingFunction()
        
        # 尝试清理旧的同名 Collection 以防损坏
        try:
            _client.delete_collection(name=collection_name)
        except:
            pass
        
        # 创建新的 Collection
        _client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=dummy_embedding_fn
        )
        print(f"DEBUG: Collection '{collection_name}' setup successfully.")
        return True
    except Exception as e:
        print(f"ERROR: Error setting up collection: {e}")
        return False

def index_data_if_needed(client, data, embedding_model):
    """检查并执行数据索引。"""
    global id_to_doc_map

    if not client or isinstance(client, bool):
        return False

    collection_name = COLLECTION_NAME
    dummy_embedding_fn = DummyEmbeddingFunction()
    
    try:
        # 获取现有的 collection
        try:
            collection = client.get_collection(name=collection_name, embedding_function=dummy_embedding_fn)
        except:
            # 如果不存在，则创建
            collection = client.get_or_create_collection(name=collection_name, embedding_function=dummy_embedding_fn)
        
        current_count = collection.count()
    except Exception as e:
        print(f"DEBUG: Collection not ready, error: {e}")
        return False

    data_to_index = data[:MAX_ARTICLES_TO_INDEX]
    needed_count = len(data_to_index)
    
    if current_count < needed_count:
        # 开始索引流程
        docs_for_embedding = []
        ids = []
        metadatas = []
        temp_id_map = {}

        for i, doc in enumerate(data_to_index):
            title = doc.get('title', '') or ""
            abstract = doc.get('abstract', '') or ""
            content = f"Title: {title}\nAbstract: {abstract}".strip()
            if not content: continue

            doc_id = str(i)
            ids.append(doc_id)
            docs_for_embedding.append(content)
            metadatas.append({
                "title": title,
                "source_file": doc.get('source_file', ''),
                "chunk_index": doc.get('chunk_index', i)
            })
            temp_id_map[i] = {'title': title, 'abstract': abstract, 'content': content}

        if docs_for_embedding:
            # 这里的 st.spinner 保留，因为这是耗时操作，需要给用户反馈
            with st.spinner(f"正在对 {len(docs_for_embedding)} 条文献进行向量化..."):
                import time
                embeddings = embedding_model.encode(docs_for_embedding)
                collection.add(
                    ids=ids,
                    embeddings=embeddings.tolist(),
                    documents=docs_for_embedding,
                    metadatas=metadatas
                )
                id_to_doc_map.update(temp_id_map)
                print(f"DEBUG: Indexed {len(ids)} documents.")
                return True
    else:
        # 如果已经索引过了，填充内存映射表
        if not id_to_doc_map:
            for i, doc in enumerate(data_to_index):
                id_to_doc_map[i] = {
                    'title': doc.get('title', ''),
                    'abstract': doc.get('abstract', ''),
                    'content': f"Title: {doc.get('title', '')}\nAbstract: {doc.get('abstract', '')}"
                }
        return True
    return False

def search_similar_documents(client, query, embedding_model, top_k=None):
    """在数据库中搜索相似文档。"""
    if not client or isinstance(client, bool) or not embedding_model:
        return [], []

    if top_k is None: top_k = TOP_K

    try:
        dummy_embedding_fn = DummyEmbeddingFunction()
        collection = client.get_collection(name=COLLECTION_NAME, embedding_function=dummy_embedding_fn)
        
        # 将查询文本转化为向量
        query_embedding = embedding_model.encode([query])[0]

        # 执行查询
        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k
        )

        if not results or not results['ids'] or not results['ids'][0]:
            return [], []

        hit_ids = [int(id) for id in results['ids'][0]]
        distances = results['distances'][0] if 'distances' in results else []
        
        # 之前这里的 st.write 已全部移除，改为控制台打印
        print(f"SEARCH DEBUG: Query='{query}', Hits={hit_ids}")
        
        return hit_ids, distances
    except Exception as e:
        print(f"ERROR: Search failed: {e}")
        return [], []