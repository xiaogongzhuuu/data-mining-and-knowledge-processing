"""
RAG 系统评估脚本
生成 eval_generation.json 评估结果
"""

import time
import json
import os
import sys

# 添加环境变量
os.environ['HF_HOME'] = './hf_cache'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'

from config import (
    DATA_FILE, EMBEDDING_MODEL_NAME, GENERATION_MODEL_NAME, TOP_K,
    MAX_ARTICLES_TO_INDEX, COLLECTION_NAME, OLLAMA_MODEL, OLLAMA_BASE_URL,
    EMBEDDING_DIM
)
from data_utils import load_data
from models import load_embedding_model, load_generation_model
from chroma_utils import get_chroma_client, setup_chroma_collection, index_data_if_needed, search_similar_documents, id_to_doc_map
from rag_core import generate_answer
from retrieval_optimizer import hybrid_search, rerank_documents, remove_duplicate_documents

# 测试查询集
TEST_QUERIES = [
    {
        "query": "吴银根的学术思想是什么？",
        "expected_keywords": ["气血阴阳", "平", "动态", "相对"],
        "description": "测试对吴银根学术思想的检索"
    },
    {
        "query": "施杞在中医外科方面有什么贡献？",
        "expected_keywords": ["中医外科", "临床", "经验"],
        "description": "测试对施杞专业领域的检索"
    },
    {
        "query": "如何治疗慢性阻塞性肺疾病？",
        "expected_keywords": ["肺病", "治疗", "辨证"],
        "description": "测试对疾病治疗方案的检索"
    },
    {
        "query": "中医调理气血的方法有哪些？",
        "expected_keywords": ["气血", "调理", "方药"],
        "description": "测试对中医调理方法的检索"
    },
    {
        "query": "肺肾两脏的关系是什么？",
        "expected_keywords": ["肺", "肾", "母子", "相生"],
        "description": "测试对脏腑关系的检索"
    },
    {
        "query": "如何理解'以平为期'的治疗原则？",
        "expected_keywords": ["平", "调和", "阴阳", "气血"],
        "description": "测试对治疗原则的理解"
    }
]

def evaluate_retrieval_relevance(query, retrieved_docs, expected_keywords):
    """评估检索相关性"""
    if not retrieved_docs:
        return 0.0, "未检索到文档"

    # 检查检索到的文档中是否包含期望的关键词
    keyword_hits = 0
    for doc in retrieved_docs:
        content = doc.get('content', '').lower()
        for keyword in expected_keywords:
            if keyword.lower() in content:
                keyword_hits += 1
                break

    relevance_score = min(keyword_hits / len(expected_keywords), 1.0)
    return relevance_score, f"命中 {keyword_hits}/{len(expected_keywords)} 个关键词"

def evaluate_generation_quality(answer, expected_keywords):
    """评估生成质量"""
    if not answer:
        return 0.0, "未生成答案"

    # 检查答案中是否包含期望的关键词
    keyword_hits = 0
    answer_lower = answer.lower()
    for keyword in expected_keywords:
        if keyword.lower() in answer_lower:
            keyword_hits += 1

    quality_score = min(keyword_hits / len(expected_keywords), 1.0)
    return quality_score, f"答案包含 {keyword_hits}/{len(expected_keywords)} 个关键词"

def run_evaluation():
    """运行评估"""
    print("="*60)
    print("RAG 系统评估")
    print("="*60)

    # 1. 初始化组件
    print("\n[1/3] 初始化系统组件...")
    start_time = time.time()

    chroma_client = get_chroma_client()
    if not chroma_client:
        print("❌ ChromaDB 客户端初始化失败")
        return

    collection_ready = setup_chroma_collection(chroma_client)
    if not collection_ready:
        print("❌ ChromaDB Collection 设置失败")
        return

    embedding_model = load_embedding_model(EMBEDDING_MODEL_NAME)
    if not embedding_model:
        print("❌ 嵌入模型加载失败")
        return

    generation_model, tokenizer = load_generation_model(GENERATION_MODEL_NAME)
    if not generation_model:
        print("❌ 生成模型加载失败")
        return

    init_time = time.time() - start_time
    print(f"✅ 系统初始化完成，耗时: {init_time:.2f} 秒")

    # 2. 加载和索引数据
    print("\n[2/3] 加载和索引数据...")
    start_time = time.time()

    pubmed_data = load_data(DATA_FILE)
    if not pubmed_data:
        print(f"❌ 无法从 {DATA_FILE} 加载数据")
        return

    print(f"   已加载 {len(pubmed_data)} 条数据")

    indexing_successful = index_data_if_needed(chroma_client, pubmed_data, embedding_model)
    if not indexing_successful:
        print("❌ 数据索引失败")
        return

    index_time = time.time() - start_time
    print(f"✅ 数据索引完成，耗时: {index_time:.2f} 秒")

    # 3. 运行评估
    print("\n[3/3] 运行评估...")

    evaluation_results = {
        "evaluation_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "system_info": {
            "embedding_model": EMBEDDING_MODEL_NAME,
            "generation_model": f"Ollama {OLLAMA_MODEL}",
            "vector_db": "ChromaDB",
            "data_file": DATA_FILE,
            "total_documents": len(pubmed_data),
            "top_k": TOP_K
        },
        "performance_metrics": {
            "initialization_time": init_time,
            "indexing_time": index_time
        },
        "test_results": []
    }

    total_retrieval_score = 0.0
    total_generation_score = 0.0
    total_response_time = 0.0

    for i, test_case in enumerate(TEST_QUERIES, 1):
        query = test_case["query"]
        expected_keywords = test_case["expected_keywords"]
        description = test_case["description"]

        print(f"\n测试 {i}/{len(TEST_QUERIES)}: {description}")

        # 执行查询
        start_time = time.time()

        # 使用混合检索
        retrieved_docs, distances = hybrid_search(chroma_client, query, embedding_model, top_k=TOP_K)

        if not retrieved_docs:
            print("❌ 检索失败：未找到相关文档")
            continue

        # 去重
        retrieved_docs = remove_duplicate_documents(retrieved_docs)

        if not retrieved_docs:
            print("❌ 检索失败：去重后无文档")
            continue

        # 生成答案
        answer = generate_answer(query, retrieved_docs, generation_model, tokenizer)

        response_time = time.time() - start_time
        total_response_time += response_time

        # 评估
        retrieval_score, retrieval_detail = evaluate_retrieval_relevance(query, retrieved_docs, expected_keywords)
        generation_score, generation_detail = evaluate_generation_quality(answer, expected_keywords)

        total_retrieval_score += retrieval_score
        total_generation_score += generation_score

        # 收集检索结果信息
        retrieved_info = []
        for j, (doc, dist) in enumerate(zip(retrieved_docs, distances)):
            doc_id = None
            for did, ddoc in id_to_doc_map.items():
                if ddoc == doc:
                    doc_id = did
                    break
            retrieved_info.append({
                "rank": j + 1,
                "doc_id": doc_id,
                "title": doc.get('title', ''),
                "source_file": doc.get('source_file', ''),
                "similarity": float(dist) if dist else None
            })

        # 保存结果
        test_result = {
            "test_id": i,
            "description": description,
            "query": query,
            "expected_keywords": expected_keywords,
            "retrieval": {
                "num_retrieved": len(retrieved_docs),
                "relevance_score": float(retrieval_score),
                "relevance_detail": retrieval_detail,
                "documents": retrieved_info
            },
            "generation": {
                "answer": answer,
                "quality_score": float(generation_score),
                "quality_detail": generation_detail,
                "answer_length": len(answer)
            },
            "performance": {
                "response_time": float(response_time)
            }
        }

        evaluation_results["test_results"].append(test_result)

        # 打印结果
        print(f"   检索相关性: {retrieval_score:.2f} ({retrieval_detail})")
        print(f"   生成质量: {generation_score:.2f} ({generation_detail})")
        print(f"   响应时间: {response_time:.2f} 秒")

    # 4. 汇总结果
    avg_retrieval_score = total_retrieval_score / len(TEST_QUERIES)
    avg_generation_score = total_generation_score / len(TEST_QUERIES)
    avg_response_time = total_response_time / len(TEST_QUERIES)

    evaluation_results["summary"] = {
        "total_tests": len(TEST_QUERIES),
        "avg_retrieval_relevance": float(avg_retrieval_score),
        "avg_generation_quality": float(avg_generation_score),
        "avg_response_time": float(avg_response_time),
        "overall_score": float((avg_retrieval_score + avg_generation_score) / 2)
    }

    # 评级
    if avg_retrieval_score >= 0.8 and avg_generation_score >= 0.8:
        evaluation_results["summary"]["rating"] = "优秀 (⭐⭐⭐)"
    elif avg_retrieval_score >= 0.7 and avg_generation_score >= 0.7:
        evaluation_results["summary"]["rating"] = "良好 (⭐⭐)"
    elif avg_retrieval_score >= 0.6 and avg_generation_score >= 0.6:
        evaluation_results["summary"]["rating"] = "中等 (⭐)"
    else:
        evaluation_results["summary"]["rating"] = "需要改进 (⚠️)"

    # 5. 保存结果
    output_file = "eval_generation.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(evaluation_results, f, ensure_ascii=False, indent=2)

    print("\n" + "="*60)
    print("评估完成！")
    print("="*60)
    print(f"\n平均检索相关性: {avg_retrieval_score:.2%}")
    print(f"平均生成质量: {avg_generation_score:.2%}")
    print(f"平均响应时间: {avg_response_time:.2f} 秒")
    print(f"总体评分: {evaluation_results['summary']['rating']}")
    print(f"\n评估结果已保存到: {output_file}")

if __name__ == "__main__":
    try:
        run_evaluation()
    except KeyboardInterrupt:
        print("\n\n评估被用户中断")
    except Exception as e:
        print(f"\n\n❌ 评估过程中发生错误: {e}")
        import traceback
        traceback.print_exc()