"""
RAG 系统性能测试脚本
测试维度：
1. 检索相关性
2. 生成质量（语义准确性、专业术语匹配）
3. 响应时间
"""

import time
import json
import sys
import os

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

def print_section(title):
    """打印分节标题"""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60 + "\n")

def test_retrieval_relevance(query, retrieved_docs, expected_keywords):
    """测试检索相关性"""
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

def test_generation_quality(answer, expected_keywords):
    """测试生成质量"""
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

def run_performance_test():
    """运行性能测试"""
    print_section("RAG 系统性能测试")

    # 1. 初始化组件
    print("1. 初始化系统组件...")
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
    print(f"✅ 系统初始化完成，耗时: {init_time:.2f} 秒\n")

    # 2. 加载和索引数据
    print("2. 加载和索引数据...")
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
    print(f"✅ 数据索引完成，耗时: {index_time:.2f} 秒\n")

    # 3. 运行测试查询
    print_section("3. 运行测试查询")

    total_retrieval_score = 0.0
    total_generation_score = 0.0
    total_response_time = 0.0

    for i, test_case in enumerate(TEST_QUERIES, 1):
        query = test_case["query"]
        expected_keywords = test_case["expected_keywords"]
        description = test_case["description"]

        print(f"\n测试 {i}/{len(TEST_QUERIES)}: {description}")
        print(f"查询: {query}")
        print(f"期望关键词: {', '.join(expected_keywords)}")

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
        retrieval_score, retrieval_detail = test_retrieval_relevance(query, retrieved_docs, expected_keywords)
        generation_score, generation_detail = test_generation_quality(answer, expected_keywords)

        total_retrieval_score += retrieval_score
        total_generation_score += generation_score

        # 打印结果
        print(f"\n📊 检索结果:")
        print(f"   - 检索到 {len(retrieved_docs)} 个文档")
        print(f"   - 相关性评分: {retrieval_score:.2f} ({retrieval_detail})")

        print(f"\n📝 生成结果:")
        print(f"   - 答案长度: {len(answer)} 字符")
        print(f"   - 质量评分: {generation_score:.2f} ({generation_detail})")
        print(f"   - 响应时间: {response_time:.2f} 秒")

        print(f"\n💬 生成的答案:")
        print(f"   {answer[:300]}...")

    # 4. 汇总结果
    print_section("4. 性能测试汇总")

    avg_retrieval_score = total_retrieval_score / len(TEST_QUERIES)
    avg_generation_score = total_generation_score / len(TEST_QUERIES)
    avg_response_time = total_response_time / len(TEST_QUERIES)

    print(f"✅ 测试完成！共测试 {len(TEST_QUERIES)} 个查询\n")
    print(f"平均检索相关性: {avg_retrieval_score:.2%}")
    print(f"平均生成质量: {avg_generation_score:.2%}")
    print(f"平均响应时间: {avg_response_time:.2f} 秒")
    print(f"系统初始化时间: {init_time:.2f} 秒")
    print(f"数据索引时间: {index_time:.2f} 秒")

    # 性能评级
    print("\n📈 性能评级:")
    if avg_retrieval_score >= 0.8 and avg_generation_score >= 0.8:
        print("   ⭐⭐⭐ 优秀 - 系统表现优秀，检索和生成质量均达到高标准")
    elif avg_retrieval_score >= 0.7 and avg_generation_score >= 0.7:
        print("   ⭐⭐ 良好 - 系统表现良好，各项指标达到预期")
    elif avg_retrieval_score >= 0.6 and avg_generation_score >= 0.6:
        print("   ⭐ 中等 - 系统表现尚可，仍有优化空间")
    else:
        print("   ⚠️ 需要改进 - 建议进一步优化检索和生成策略")

    # 优化建议
    print("\n💡 优化建议:")
    if avg_retrieval_score < 0.7:
        print("   - 检索相关性较低，建议：")
        print("     * 增加检索的文档数量（TOP_K）")
        print("     * 优化嵌入模型或尝试其他模型")
        print("     * 改进数据分块策略")
    
    if avg_generation_score < 0.7:
        print("   - 生成质量较低，建议：")
        print("     * 进一步优化提示词工程")
        print("     * 增加上下文长度限制")
        print("     * 调整生成参数（temperature, top_p）")
    
    if avg_response_time > 10:
        print("   - 响应时间较长，建议：")
        print("     * 优化模型加载和缓存")
        print("     * 考虑使用更轻量的模型")
        print("     * 优化检索算法效率")

if __name__ == "__main__":
    try:
        run_performance_test()
    except KeyboardInterrupt:
        print("\n\n测试被用户中断")
    except Exception as e:
        print(f"\n\n❌ 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()