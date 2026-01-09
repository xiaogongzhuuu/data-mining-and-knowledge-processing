import streamlit as st
import time
import os

# 环境变量设置
os.environ['HF_HOME'] = './hf_cache'
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'

# 导入配置和工具模块
from config import (
    DATA_FILE, EMBEDDING_MODEL_NAME, TOP_K,
    MILVUS_LITE_DATA_PATH, COLLECTION_NAME,
    EMBEDDING_DIM, OLLAMA_MODEL
)
from data_utils import load_data
from models import load_embedding_model, load_generation_model
from chroma_utils import get_chroma_client, setup_chroma_collection, index_data_if_needed
from rag_core import generate_answer
from retrieval_optimizer import hybrid_search

# --- 1. 页面配置与 CSS 美化 ---
st.set_page_config(
    page_title="智能中医医疗助手",
    layout="wide",
    page_icon="🏥",
    initial_sidebar_state="expanded"
)

def inject_custom_css():
    st.markdown("""
    <style>
        /* 全局背景 */
        .main { background-color: #f8fafc; }
        
        /* 标题样式 */
        .main-header {
            text-align: center;
            padding: 2rem 0;
            background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 100%);
            color: white;
            border-radius: 15px;
            margin-bottom: 2rem;
        }

        /* 搜索框美化 */
        .stTextInput>div>div>input {
            border-radius: 10px;
            border: 2px solid #e2e8f0;
            padding: 10px 15px;
        }

        /* 按钮美化 */
        .stButton>button {
            width: 100%;
            border-radius: 10px;
            background-color: #10b981;
            color: white;
            font-weight: bold;
            height: 3rem;
            transition: all 0.3s;
        }
        .stButton>button:hover {
            background-color: #059669;
            transform: translateY(-2px);
        }

        /* 医疗文献卡片 */
        .medical-card {
            background-color: white;
            padding: 20px;
            border-radius: 12px;
            border-left: 6px solid #10b981;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            margin-bottom: 20px;
            border-top: 1px solid #f1f5f9;
        }
        
        .source-tag {
            display: inline-block;
            background-color: #dbeafe;
            color: #1e40af;
            font-size: 0.75rem;
            padding: 2px 10px;
            border-radius: 20px;
            font-weight: 600;
            margin-bottom: 10px;
        }

        /* AI 回答区域 */
        .answer-box {
            background-color: #ffffff;
            padding: 30px;
            border-radius: 15px;
            border: 1px solid #e2e8f0;
            border-top: 6px solid #3b82f6;
            line-height: 1.8;
            color: #1e293b;
            font-size: 1.05rem;
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05);
        }

        /* 隐藏 Streamlit 默认页脚 */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

inject_custom_css()

# --- 2. 初始化核心后端 ---
embedding_loaded = False
generation_loaded = False
collection_is_ready = False

# 侧边栏状态
with st.sidebar:
    st.markdown("### 🏥 系统控制面板")
    st.image("https://cdn-icons-png.flaticon.com/512/387/387561.png", width=80)
    st.divider()
    status_container = st.container()

# 1. 加载模型
embedding_model = load_embedding_model(EMBEDDING_MODEL_NAME)
embedding_loaded = embedding_model is not None

generation_model, tokenizer = load_generation_model(OLLAMA_MODEL)
generation_loaded = generation_model is not None

# 2. 初始化数据库 (修正关键点)
client = get_chroma_client()
if client:
    # 这里的 success 是布尔值
    setup_success = setup_chroma_collection(client)
    if setup_success:
        raw_data = load_data(DATA_FILE)
        if raw_data:
            with st.spinner("同步向量数据库中..."):
                # 注意：这里传的是 client 对象，而不是上面的布尔值
                collection_is_ready = index_data_if_needed(client, raw_data, embedding_model)

# 更新侧边栏状态显示
with status_container:
    col_s1, col_s2 = st.columns(2)
    col_s1.metric("向量引擎", "Chroma", delta="就绪" if collection_is_ready else "异常")
    col_s2.metric("生成模型", "Qwen-8B", delta="就绪" if generation_loaded else "未连接", 
                  delta_color="normal" if generation_loaded else "inverse")
# --- 3. 主界面布局 ---
st.markdown("""
    <div class="main-header">
        <h1>智能中医医疗问答系统</h1>
        <p>专业 · 安全 · 离线知识库检索</p>
    </div>
""", unsafe_allow_html=True)

# 搜索区
col_left, col_mid, col_right = st.columns([1, 8, 1])
with col_mid:
    query = st.text_input("", placeholder="请输入症状、药物或中医养生问题（例如：如何缓解夏季感冒？）")
    search_button = st.button("🔍 开始深度检索与分析")

# --- 4. 检索与生成逻辑 ---
if search_button:
    if not query:
        st.warning("⚠️ 请先输入您的问题。")
    elif not collection_is_ready:
        st.error("❌ 数据库未准备就绪，请检查本地数据文件。")
    else:
        start_time = time.time()
        
        with st.status("🚀 正在处理您的请求...", expanded=True) as status:
            # 执行检索
            status.write("正在从 500+ 中医文献中检索相关信息...")
            retrieved_docs, distances = hybrid_search(query, client, embedding_model, top_k_val)
            
            if not retrieved_docs:
                status.update(label="❌ 未能找到相关参考资料", state="error")
                st.error("抱歉，我们的知识库中目前没有关于此问题的记录。")
            else:
                status.write(f"成功命中 {len(retrieved_docs)} 条高质量文献。")
                
                # 分页展示结果
                tab_ans, tab_ref = st.tabs(["✨ AI 深度分析", "📖 原始文献参考"])
                
                with tab_ans:
                    if generation_loaded:
                        status.update(label="正在组织语言并生成专业建议...", state="running")
                        with st.spinner("AI 医师正在阅读文献..."):
                            answer = generate_answer(query, retrieved_docs, generation_model, tokenizer)
                            st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)
                    else:
                        st.info("💡 当前处于“仅搜索模式”。若要启用 AI 自动回答，请在本地启动 Ollama 服务。")
                        st.markdown("### 检索到的关键信息预览：")
                        st.write(retrieved_docs[0].get('content', '')[:500] + "...")

                with tab_ref:
                    for i, doc in enumerate(retrieved_docs):
                        st.markdown(f"""
                        <div class="medical-card">
                            <span class="source-tag">来源：{doc.get('source_file', '传统医学典籍')}</span>
                            <h4 style="margin:0 0 10px 0; color:#1e293b;">{doc.get('title', '医疗条目')}</h4>
                            <p style="color:#475569; font-size:0.95rem;">{doc.get('content', '')}</p>
                            <hr style="border:0; border-top:1px solid #f1f5f9; margin:10px 0;">
                            <small style="color:#94a3b8;">关联度评分: {1/(1+distances[i]):.4f}</small>
                        </div>
                        """, unsafe_allow_html=True)

                end_time = time.time()
                status.update(label=f"✅ 处理完成 (用时: {end_time - start_time:.2f}s)", state="complete", expanded=False)

# 页脚
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #94a3b8; font-size: 0.8rem;'>"
    "声明：本系统提供的回答仅供参考，不作为医学诊断依据。如遇不适请及时就医。"
    "</div>", 
    unsafe_allow_html=True
)