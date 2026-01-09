import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
import os
import sys

# Import config variables including the global map
from config import (
    MILVUS_LITE_DATA_PATH, COLLECTION_NAME, EMBEDDING_DIM,
    MAX_ARTICLES_TO_INDEX, TOP_K, id_to_doc_map, EMBEDDING_MODEL_NAME
)

class DummyEmbeddingFunction(embedding_functions.EmbeddingFunction):
    """
    虚拟嵌入函数 - 这只是为了让 ChromaDB 不尝试下载默认模型。
    实际的嵌入向量会在 index_data_if_needed 中预先计算。
    """
    def __call__(self, input):
        """
        返回虚拟嵌入（全零向量）。
        这个函数不会被真正调用，因为我们总是提供预计算的嵌入。
        """
        if isinstance(input, str):
            input = [input]
        # 返回虚拟向量列表
        return [[0.0] * EMBEDDING_DIM for _ in input]

def get_chroma_client():
    """Initializes and returns a ChromaDB client."""
    try:
        st.write(f"Initializing ChromaDB client...")
        persist_dir = "./chroma_data"
        if not os.path.exists(persist_dir):
            os.makedirs(persist_dir, exist_ok=True)
        
        # 支持多个 chromadb 版本：优先使用 1.4.x 的 PersistentClient；否则回退到旧版 Client
        client = None
        if hasattr(chromadb, "PersistentClient"):
            st.write("Using chromadb.PersistentClient()")
            client = chromadb.PersistentClient(path=persist_dir)
        else:
            st.write("PersistentClient not found, trying chromadb.Client(...) fallback")
            try:
                # 在旧版本中通过 settings 启用持久化（duckdb+parquet）
                settings = chromadb.config.Settings()
                # 覆盖 persist_directory
                try:
                    settings.persist_directory = persist_dir
                except Exception:
                    # 某些版本的 Settings 可能不允许直接赋值，通过构造函数传参也可
                    settings = chromadb.config.Settings(persist_directory=persist_dir)
                # 如果存在 chroma_db_impl，设置为 duckdb+parquet 以启用持久化
                try:
                    settings.chroma_db_impl = "duckdb+parquet"
                except Exception:
                    pass
                client = chromadb.Client(settings)
            except Exception as e:
                # 如果回退也失败，抛出以进入外层 except
                raise
        
        st.success(f"ChromaDB client initialized! Data persisted to '{persist_dir}'")
        return client
    except Exception as e:
        st.error(f"Failed to initialize ChromaDB client: {e}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
        return None

def setup_chroma_collection(_client):
    """Ensures the specified collection exists in ChromaDB, completely offline."""
    if not _client:
        st.error("ChromaDB client not available.")
        return False
    try:
        collection_name = COLLECTION_NAME
        st.write(f"Setting up ChromaDB collection '{collection_name}'...")

        # 首先尝试删除可能损坏的现有 collection
        try:
            _client.delete_collection(name=collection_name)
            st.write(f"Removed existing collection '{collection_name}'.")
        except Exception as e:
            st.write(f"No existing collection to remove (expected): {str(e)[:100]}")
        
        # 创建虚拟嵌入函数，以阻止 ChromaDB 自动下载模型
        dummy_embedding_fn = DummyEmbeddingFunction()
        
        # 创建全新的 collection，使用虚拟嵌入函数
        try:
            collection = _client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
                embedding_function=dummy_embedding_fn,
                get_or_create=False  # 确保创建新的
            )
            st.success(f"✅ Collection '{collection_name}' created successfully.")
        except Exception as create_error:
            # 如果创建失败，可能是因为已存在，尝试获取
            st.write(f"Create failed, attempting to get existing collection: {create_error}")
            try:
                collection = _client.get_collection(
                    name=collection_name,
                    embedding_function=dummy_embedding_fn
                )
                st.success(f"✅ Found existing collection '{collection_name}'.")
            except Exception as get_error:
                st.error(f"❌ Failed to create or get collection: {get_error}")
                import traceback
                st.error(f"Traceback: {traceback.format_exc()}")
                return False

        # 验证 collection 是否可用
        try:
            count = collection.count()
            st.write(f"Collection '{collection_name}' ready. Current document count: {count}")
        except Exception as count_error:
            st.warning(f"Could not verify collection count: {count_error}")

        return True

    except Exception as e:
        st.error(f"Error setting up ChromaDB collection '{COLLECTION_NAME}': {e}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
        return False


def index_data_if_needed(client, data, embedding_model):
    """Checks if data needs indexing and performs it using ChromaDB."""
    global id_to_doc_map

    if not client:
        st.error("ChromaDB client not available for indexing.")
        return False

    collection_name = COLLECTION_NAME
    
    try:
        # 创建虚拟嵌入函数
        dummy_embedding_fn = DummyEmbeddingFunction()
        
        # 获取或创建 collection，使用虚拟嵌入函数
        try:
            collection = client.get_collection(
                name=collection_name,
                embedding_function=dummy_embedding_fn
            )
        except:
            # 如果获取失败，尝试创建
            collection = client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
                embedding_function=dummy_embedding_fn
            )
        
        current_count = collection.count()
    except Exception as e:
        st.write(f"Could not retrieve collection, attempting to setup. Error: {e}")
        if not setup_chroma_collection(client):
            return False
        try:
            dummy_embedding_fn = DummyEmbeddingFunction()
            collection = client.get_collection(
                name=collection_name,
                embedding_function=dummy_embedding_fn
            )
            current_count = collection.count()
        except Exception as get_error:
            st.error(f"Failed to get collection after setup: {get_error}")
            import traceback
            st.error(f"Traceback: {traceback.format_exc()}")
            return False

    st.write(f"Entities currently in ChromaDB collection '{collection_name}': {current_count}")

    data_to_index = data[:MAX_ARTICLES_TO_INDEX]
    needed_count = len(data_to_index)
    
    if current_count < needed_count:
        st.warning(f"Indexing required ({current_count}/{needed_count} documents found). This may take a while...")

        # Prepare data for indexing
        docs_for_embedding = []
        ids = []
        metadatas = []
        temp_id_map = {}

        with st.spinner("Preparing data for indexing..."):
            for i, doc in enumerate(data_to_index):
                title = doc.get('title', '') or ""
                abstract = doc.get('abstract', '') or ""
                content = f"Title: {title}\nAbstract: {abstract}".strip()
                if not content:
                    continue

                doc_id = str(i)  # ChromaDB expects string IDs
                ids.append(doc_id)
                docs_for_embedding.append(content)
                metadatas.append({
                    "title": title,
                    "source_file": doc.get('source_file', ''),
                    "chunk_index": doc.get('chunk_index', i)
                })
                temp_id_map[i] = {
                    'title': title,
                    'abstract': abstract,
                    'content': content
                }

        if docs_for_embedding:
            st.write(f"Embedding {len(docs_for_embedding)} documents...")
            with st.spinner("Generating embeddings..."):
                import time
                start_embed = time.time()
                embeddings = embedding_model.encode(docs_for_embedding, show_progress_bar=True)
                end_embed = time.time()
                st.write(f"Embedding took {end_embed - start_embed:.2f} seconds.")

            st.write("Inserting data into ChromaDB...")
            with st.spinner("Inserting..."):
                try:
                    start_insert = time.time()
                    # 传入预计算的 embeddings
                    collection.add(
                        ids=ids,
                        embeddings=embeddings.tolist(),
                        documents=docs_for_embedding,
                        metadatas=metadatas
                    )
                    end_insert = time.time()
                    st.success(f"Successfully indexed {len(ids)} documents. Insert took {end_insert - start_insert:.2f} seconds.")
                    id_to_doc_map.update(temp_id_map)
                    return True
                except Exception as e:
                    st.error(f"Error inserting data into ChromaDB: {e}")
                    import traceback
                    st.error(f"Traceback: {traceback.format_exc()}")
                    return False
        else:
            st.error("No valid text content found in the data to index.")
            return False
    else:
        st.write("Data count suggests indexing is complete.")
        # Populate the global map if it's empty but indexing isn't needed
        if not id_to_doc_map:
            for i, doc in enumerate(data_to_index):
                title = doc.get('title', '') or ""
                abstract = doc.get('abstract', '') or ""
                content = f"Title: {title}\nAbstract: {abstract}".strip()
                id_to_doc_map[i] = {
                    'title': title,
                    'abstract': abstract,
                    'content': content
                }
        return True


def search_similar_documents(client, query, embedding_model, top_k=None):
    """Searches ChromaDB for documents similar to the query."""
    if not client or not embedding_model:
        st.error("ChromaDB client or embedding model not available for search.")
        return [], []

    # 如果没有指定 top_k，使用配置中的默认值
    if top_k is None:
        top_k = TOP_K

    collection_name = COLLECTION_NAME
    try:
        # 创建虚拟嵌入函数
        dummy_embedding_fn = DummyEmbeddingFunction()
        collection = client.get_collection(
            name=collection_name,
            embedding_function=dummy_embedding_fn
        )
        query_embedding = embedding_model.encode([query])[0]

        # Search in ChromaDB
        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k
        )

        # 调试信息
        st.write(f"Query: {query}")
        st.write(f"Top K: {top_k}")
        st.write(f"Results keys: {results.keys()}")
        if 'ids' in results:
            st.write(f"Found {len(results['ids'][0]) if results['ids'] else 0} results")

        # Process results
        if not results or not results['ids'] or not results['ids'][0]:
            return [], []

        hit_ids = [int(id) for id in results['ids'][0]]
        distances = results['distances'][0] if 'distances' in results else []
        st.write(f"Hit IDs: {hit_ids}")
        st.write(f"Distances: {distances}")
        return hit_ids, distances
    except Exception as e:
        st.error(f"Error during ChromaDB search: {e}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
        return [], []