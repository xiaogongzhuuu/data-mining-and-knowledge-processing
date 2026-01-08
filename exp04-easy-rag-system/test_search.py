"""
检索测试脚本
测试检索功能是否正常工作
"""

import os
import sys

# 设置环境变量
os.environ['HF_HOME'] = './hf_cache'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'

from config import (
    DATA_FILE, EMBEDDING_MODEL_NAME, TOP_K, COLLECTION_NAME
)
from data_utils import load_data
from models import load_embedding_model
from chroma_utils import get_chroma_client, setup_chroma_collection, id_to_doc_map
from retrieval_optimizer import hybrid_search

def test_search():
    """测试检索功能"""
    print("="*60)
    print("检索功能测试")
    print("="*60)

    # 1. 初始化组件
    print("\n1. 初始化组件...")
    client = get_chroma_client()
    if not client:
        print("ChromaDB 客户端初始化失败")
        return

    collection_ready = setup_chroma_collection(client)
    if not collection_ready:
        print("Collection 设置失败")
        return

    embedding_model = load_embedding_model(EMBEDDING_MODEL_NAME)
    if not embedding_model:
        print("嵌入模型加载失败")
        return

    print("组件初始化成功\n")

    # 2. 填充 id_to_doc_map
    print("2. 加载文档映射...")
    pubmed_data = load_data(DATA_FILE)
    if pubmed_data and not id_to_doc_map:
        for i, doc in enumerate(pubmed_data):
            title = doc.get('title', '') or ""
            abstract = doc.get('abstract', '') or ""
            content = f"Title: {title}\nAbstract: {abstract}".strip()
            id_to_doc_map[i] = {
                'title': title,
                'abstract': abstract,
                'content': content,
                'source_file': doc.get('source_file', '')
            }
    print(f"文档映射加载完成，共 {len(id_to_doc_map)} 个文档\n")

    # 3. 测试查询
    test_queries = [
        "感冒了咋办",
        "吴银根的学术思想有哪些",
        "如何治疗哮喘",
        "什么是肺络痹阻"
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*60}")
        print(f"测试查询 {i}: {query}")
        print('='*60)

        # 检索
        docs, distances = hybrid_search(client, query, embedding_model, top_k=TOP_K)

        if not docs:
            print("未检索到相关文档")
            continue

        print(f"\n检索到 {len(docs)} 个文档:\n")

        for j, (doc, dist) in enumerate(zip(docs, distances), 1):
            print(f"文档 {j}:")
            print(f"  标题: {doc.get('title', '未知')}")
            print(f"  来源: {doc.get('source_file', '未知')}")
            print(f"  相似度距离: {dist:.4f}")
            print(f"  内容预览: {doc.get('content', '')[:100]}...")
            print()

    print("="*60)
    print("测试完成")
    print("="*60)

if __name__ == "__main__":
    try:
        test_search()
    except Exception as e:
        print(f"\n测试失败: {e}")
        import traceback
        traceback.print_exc()
