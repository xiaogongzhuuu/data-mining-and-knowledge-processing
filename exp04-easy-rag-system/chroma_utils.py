import streamlit as st
import chromadb
from chromadb.config import Settings
import os

# Import config variables including the global map
from config import (
    MILVUS_LITE_DATA_PATH, COLLECTION_NAME, EMBEDDING_DIM,
    MAX_ARTICLES_TO_INDEX, TOP_K, id_to_doc_map
)

def get_chroma_client():
    """Initializes and returns a ChromaDB client."""
    try:
        st.write(f"Initializing ChromaDB client...")
        # Use current directory for persistent storage
        client = chromadb.PersistentClient(path=".")
        st.success("ChromaDB client initialized!")
        return client
    except Exception as e:
        st.error(f"Failed to initialize ChromaDB client: {e}")
        return None

def setup_chroma_collection(_client):
    """Ensures the specified collection exists in ChromaDB."""
    if not _client:
        st.error("ChromaDB client not available.")
        return False
    try:
        collection_name = COLLECTION_NAME

        # Try to get existing collection
        try:
            collection = _client.get_collection(name=collection_name)
            st.write(f"Found existing collection: '{collection_name}'.")

            # 检查是否使用余弦距离
            metadata = collection.metadata
            if metadata and metadata.get("hnsw:space") != "cosine":
                st.warning(f"Collection uses '{metadata.get('hnsw:space')}' distance, should use 'cosine'.")
                st.warning("Please delete chroma.sqlite3 and restart.")
        except:
            # Collection doesn't exist, create it
            st.write(f"Collection '{collection_name}' not found. Creating...")
            collection = _client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}  # Use cosine similarity
            )
            st.write(f"Collection '{collection_name}' created with cosine distance.")

        # Get current count
        count = collection.count()
        st.write(f"Collection '{collection_name}' ready. Current entity count: {count}")

        return True

    except Exception as e:
        st.error(f"Error setting up ChromaDB collection '{COLLECTION_NAME}': {e}")
        return False


def index_data_if_needed(client, data, embedding_model):
    """Checks if data needs indexing and performs it using ChromaDB."""
    global id_to_doc_map

    if not client:
        st.error("ChromaDB client not available for indexing.")
        return False

    collection_name = COLLECTION_NAME
    
    try:
        collection = client.get_collection(name=collection_name)
        current_count = collection.count()
    except:
        st.write(f"Could not retrieve collection, attempting to setup.")
        if not setup_chroma_collection(client):
            return False
        collection = client.get_collection(name=collection_name)
        current_count = 0

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
        collection = client.get_collection(name=collection_name)
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
        return [], []