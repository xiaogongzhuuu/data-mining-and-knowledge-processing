import streamlit as st
import time
import os
os.environ['HF_HOME'] = './hf_cache'
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'


# Import functions and config from other modules
from config import (
    DATA_FILE, EMBEDDING_MODEL_NAME, GENERATION_MODEL_NAME, TOP_K,
    MAX_ARTICLES_TO_INDEX, MILVUS_LITE_DATA_PATH, COLLECTION_NAME,
    id_to_doc_map, OLLAMA_MODEL # Import the global map and ollama config
)
from data_utils import load_data
from models import load_embedding_model, load_generation_model
# Import ChromaDB functions
from chroma_utils import get_chroma_client, setup_chroma_collection, index_data_if_needed
from rag_core import generate_answer
# Import optimization modules
from retrieval_optimizer import hybrid_search, remove_duplicate_documents

# --- Streamlit UI 设置 ---
st.set_page_config(layout="wide")
st.title("📄 医疗 RAG 系统 (ChromaDB + Ollama)")
st.markdown(f"使用 ChromaDB, `{EMBEDDING_MODEL_NAME}`, 和 Ollama `{OLLAMA_MODEL}`。")

# --- 初始化变量（确保在所有代码路径中都有定义）---
embedding_loaded = False
generation_loaded = False
collection_is_ready = False

# --- 初始化与缓存 ---
# 获取 ChromaDB 客户端 (如果未缓存则初始化)
chroma_client = get_chroma_client()

if chroma_client:
    # 设置 collection (如果未缓存则创建/加载索引)
    collection_is_ready = setup_chroma_collection(chroma_client)

    # 加载模型 (缓存)
    embedding_model = load_embedding_model(EMBEDDING_MODEL_NAME)
    generation_model, tokenizer = load_generation_model(GENERATION_MODEL_NAME)

    # 检查组件是否成功加载
    # embedding_model 必须可用，generation_model 可选（可以在搜索模式下工作）
    embedding_loaded = embedding_model is not None
    generation_loaded = generation_model is not None
    
    if collection_is_ready and embedding_loaded:
        # 加载数据 (未缓存)
        pubmed_data = load_data(DATA_FILE)

        # 如果需要则索引数据 (这会填充 id_to_doc_map)
        if pubmed_data:
            indexing_successful = index_data_if_needed(chroma_client, pubmed_data, embedding_model)
        else:
            st.warning(f"无法从 {DATA_FILE} 加载数据。跳过索引。")
            indexing_successful = False # 如果没有数据，则视为不成功

        st.divider()

        # --- RAG 交互部分 ---
        if not indexing_successful and not id_to_doc_map:
             st.error("数据索引失败或不完整，且没有文档映射。RAG 功能已禁用。")
        else:
            # 显示当前模式信息
            if generation_loaded:
                st.success("✅ 系统处于完整 RAG 模式（搜索 + 生成）")
            else:
                st.info("🔍 系统处于搜索模式（仅检索，生成功能不可用）")
            
            query = st.text_input("请提出关于已索引医疗文章的问题:", key="query_input")

            if st.button("搜索", key="submit_button") and query:
                start_time = time.time()

                # 1. 使用混合检索搜索相关文档
                with st.spinner("正在搜索相关文档..."):
                    retrieved_docs, distances = hybrid_search(
                        query, chroma_client, embedding_model, top_k=TOP_K
                    )

                if not retrieved_docs:
                    st.warning("在数据库中找不到相关文档。")
                else:
                    # 2. 去重
                    retrieved_docs = remove_duplicate_documents(retrieved_docs)

                    if not retrieved_docs:
                        st.error("检索结果为空。")
                    else:
                        st.subheader("检索到的上下文文档:")
                        for i, doc in enumerate(retrieved_docs):
                            # 如果距离可用则显示，否则只显示 ID
                            dist_str = f", 相似度: {distances[i]:.4f}" if distances and i < len(distances) else ""
                            # 从 id_to_doc_map 中查找文档 ID
                            doc_id = None
                            for did, ddoc in id_to_doc_map.items():
                                if ddoc == doc:
                                    doc_id = did
                                    break
                            doc_id_str = f" [ID: {doc_id}]" if doc_id is not None else ""
                            with st.expander(f"文档 {i+1}{dist_str}{doc_id_str} - {doc['title'][:60]}"):
                                st.write(f"**文档 ID:** {doc_id if doc_id is not None else '未知'}")
                                st.write(f"**标题:** {doc['title']}")
                                st.write(f"**来源:** {doc.get('source_file', '未知')}")
                                st.write(f"**摘要:** {doc['abstract'][:500]}...")

                        st.divider()

                        # 3. 生成答案（如果生成模型可用）
                        if generation_loaded:
                            st.subheader("生成的答案:")
                            with st.spinner("正在根据上下文生成答案..."):
                                answer = generate_answer(query, retrieved_docs, generation_model, tokenizer)
                                st.write(answer)
                        else:
                            st.info("💡 生成功能不可用。请启动 Ollama 服务以启用答案生成功能。")

                end_time = time.time()
                st.info(f"总耗时: {end_time - start_time:.2f} 秒")

    else:
        if not embedding_loaded:
            st.error("❌ 嵌入模型加载失败。无法继续。")
        if not collection_is_ready:
            st.error("❌ ChromaDB collection 设置失败。请检查日志。")
else:
    st.error("❌ 初始化 ChromaDB 客户端失败。请检查日志。")


# --- 页脚/信息侧边栏 ---
st.sidebar.header("系统配置")
st.sidebar.markdown(f"**向量存储:** ChromaDB")
st.sidebar.markdown(f"**数据路径:** `{MILVUS_LITE_DATA_PATH}`")
st.sidebar.markdown(f"**Collection:** `{COLLECTION_NAME}`")
st.sidebar.markdown(f"**数据文件:** `{DATA_FILE}`")
st.sidebar.markdown(f"**嵌入模型:** `{EMBEDDING_MODEL_NAME}`")
st.sidebar.markdown(f"**生成模型:** Ollama `{OLLAMA_MODEL}`")
st.sidebar.markdown(f"**最大索引数:** `{MAX_ARTICLES_TO_INDEX}`")
st.sidebar.markdown(f"**检索 Top K:** `{TOP_K}`")

st.sidebar.header("模式信息")
if generation_loaded:
    st.sidebar.success("✅ 完整 RAG 模式")
else:
    st.sidebar.warning("🔍 搜索模式（无生成）")
    st.sidebar.info("启动 Ollama: `ollama serve`")