"""
快速验证脚本
快速检查RAG系统的关键指标
"""

import os
import sys

os.environ['HF_HOME'] = './hf_cache'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'

from config import TOP_K
from models import load_embedding_model
from chroma_utils import get_chroma_client, setup_chroma_collection, id_to_doc_map
from data_utils import load_data
from retrieval_optimizer import hybrid_search

def quick_validation():
    """快速验证"""
    print("="*60)
    print("RAG 系统快速验证")
    print("="*60)

    # 1. 检查模型
    print("\n[1/5] 检查模型...")
    try:
        model = load_embedding_model('moka-ai/m3e-base')
        if model:
            print("  [OK] 模型加载成功")
            print(f"  维度: {model.get_sentence_embedding_dimension()}")
        else:
            print("  [FAILED] 模型加载失败")
            return False
    except Exception as e:
        print(f"  [FAILED] {e}")
        return False

    # 2. 检查数据库
    print("\n[2/5] 检查数据库...")
    try:
        client = get_chroma_client()
        if not client:
            print("  [FAILED] ChromaDB 客户端初始化失败")
            return False

        collection_ready = setup_chroma_collection(client)
        if not collection_ready:
            print("  [FAILED] Collection 设置失败")
            return False

        print("  [OK] 数据库正常")
    except Exception as e:
        print(f"  [FAILED] {e}")
        return False

    # 3. 检查数据
    print("\n[3/5] 检查数据...")
    try:
        pubmed_data = load_data("./data/processed_data.json")
        if pubmed_data:
            print(f"  [OK] 数据加载成功，共 {len(pubmed_data)} 个文档")
        else:
            print("  [FAILED] 数据加载失败")
            return False
    except Exception as e:
        print(f"  [FAILED] {e}")
        return False

    # 4. 填充文档映射
    print("\n[4/5] 准备文档映射...")
    if not id_to_doc_map:
        for i, doc in enumerate(pubmed_data):
            title = doc.get('title', '')
            abstract = doc.get('abstract', '')
            content = f"Title: {title}\nAbstract: {abstract}".strip()
            id_to_doc_map[i] = {
                'title': title,
                'abstract': abstract,
                'content': content,
                'source_file': doc.get('source_file', '')
            }
        print(f"  [OK] 文档映射完成，共 {len(id_to_doc_map)} 个文档")

    # 5. 测试检索
    print("\n[5/5] 测试检索...")
    test_query = "吴银根的学术思想"
    print(f"  查询: {test_query}")

    try:
        docs, distances = hybrid_search(client, test_query, model, top_k=TOP_K)

        if not docs:
            print("  [FAILED] 未检索到文档")
            return False

        print(f"  [OK] 检索到 {len(docs)} 个文档")

        # 检查距离
        print(f"\n  距离检查:")
        for i, (doc, dist) in enumerate(zip(docs, distances), 1):
            print(f"    文档 {i}: {dist:.4f}")

            # 距离应该在 0-2 之间（余弦距离）
            if dist > 2:
                print(f"      [WARNING] 距离过大，可能不是余弦距离")
                print(f"      建议: 删除 chroma.sqlite3 并重新启动系统")

        # 检查文档多样性
        titles = [doc.get('title', '') for doc in docs]
        unique_titles = set(titles)
        if len(unique_titles) < len(titles):
            print(f"\n  [WARNING] 检索结果缺乏多样性")
            print(f"  检索到 {len(docs)} 个文档，但只有 {len(unique_titles)} 个不同的标题")
        else:
            print(f"\n  [OK] 检索结果有良好的多样性")

    except Exception as e:
        print(f"  [FAILED] {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "="*60)
    print("验证完成")
    print("="*60)
    return True

if __name__ == "__main__":
    success = quick_validation()
    sys.exit(0 if success else 1)
