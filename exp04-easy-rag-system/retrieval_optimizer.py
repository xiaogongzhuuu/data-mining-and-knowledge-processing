"""
检索优化模块
提供重排序、混合检索、查询扩展等高级检索功能
"""

import re
from typing import List, Dict, Tuple
from collections import Counter
import numpy as np

def rerank_documents(query: str, documents: List[Dict], distances: List[float], 
                     embedding_model, top_k: int = 3) -> Tuple[List[Dict], List[float]]:
    """
    重排序检索到的文档，结合语义相似度和关键词匹配
    
    Args:
        query: 用户查询
        documents: 检索到的文档列表
        distances: 原始距离分数
        embedding_model: 嵌入模型
        top_k: 返回的文档数量
    
    Returns:
        重排序后的文档列表和距离
    """
    if not documents:
        return [], []
    
    # 提取查询关键词
    query_keywords = extract_keywords(query)
    
    # 为每个文档计算综合分数
    reranked_scores = []
    for i, doc in enumerate(documents):
        # 1. 语义相似度分数（归一化到0-1）
        semantic_score = 1.0 / (1.0 + distances[i]) if distances[i] > 0 else 1.0
        
        # 2. 关键词匹配分数
        keyword_score = calculate_keyword_match(query_keywords, doc.get('content', ''))
        
        # 3. 文档长度惩罚（避免过长的文档）
        length_penalty = 1.0 / (1.0 + len(doc.get('content', '')) / 2000.0)
        
        # 综合分数（可调整权重）
        combined_score = 0.6 * semantic_score + 0.3 * keyword_score + 0.1 * length_penalty
        reranked_scores.append((combined_score, i))
    
    # 按综合分数排序
    reranked_scores.sort(reverse=True, key=lambda x: x[0])
    
    # 返回top_k个文档
    result_docs = []
    result_distances = []
    for score, idx in reranked_scores[:top_k]:
        result_docs.append(documents[idx])
        result_distances.append(distances[idx])
    
    return result_docs, result_distances

def extract_keywords(text: str, top_n: int = 10) -> List[str]:
    """
    从文本中提取关键词（改进版，支持中文）

    Args:
        text: 输入文本
        top_n: 返回的关键词数量

    Returns:
        关键词列表
    """
    # 中文停用词
    stop_words = {'的', '是', '在', '了', '和', '与', '或', '但', '不', '这', '那', '有', '我', '你', '他', '她', '它', '们',
                  '什么', '怎么', '如何', '哪些', '哪个', '吗', '呢', '啊', '吧', '呀', '嘛', '呗', '哈', '啦',
                  '一个', '一些', '一种', '这个', '那个', '这些', '那些', '哪些', '怎样', '怎么', '如何'}

    # 移除标点符号
    text = re.sub(r'[^\w\u4e00-\u9fff]', ' ', text)

    # 提取2-4个字的词组（中医术语通常是2-4个字）
    keywords = []
    for length in [4, 3, 2]:  # 优先提取长词
        for i in range(len(text) - length + 1):
            word = text[i:i+length]
            # 过滤条件：不是停用词、长度>=2、包含中文
            if word not in stop_words and len(word) >= 2 and any('\u4e00' <= c <= '\u9fff' for c in word):
                keywords.append(word)

    # 统计词频
    word_freq = Counter(keywords)

    # 返回频率最高的词
    return [word for word, _ in word_freq.most_common(top_n)]

def calculate_keyword_match(query_keywords: List[str], doc_content: str) -> float:
    """
    计算关键词匹配分数
    
    Args:
        query_keywords: 查询关键词列表
        doc_content: 文档内容
    
    Returns:
        匹配分数（0-1）
    """
    if not query_keywords:
        return 0.0
    
    doc_content_lower = doc_content.lower()
    matched = sum(1 for keyword in query_keywords if keyword.lower() in doc_content_lower)
    
    return matched / len(query_keywords)

def hybrid_search(query: str, chroma_client, embedding_model,
                  top_k: int = 5) -> Tuple[List[Dict], List[float]]:
    """
    混合检索：结合向量检索和关键词检索

    Args:
        query: 用户查询
        chroma_client: ChromaDB 客户端
        embedding_model: 嵌入模型
        top_k: 返回的文档数量

    Returns:
        检索到的文档列表和距离
    """
    from chroma_utils import search_similar_documents, id_to_doc_map
    from config import TOP_K

    # 1. 向量检索（检索更多候选文档）
    retrieval_top_k = min(top_k * 2, TOP_K * 2)  # 检索两倍的候选文档
    retrieved_ids, distances = search_similar_documents(chroma_client, query, embedding_model, top_k=retrieval_top_k)
    
    if not retrieved_ids:
        return [], []
    
    # 获取文档内容
    retrieved_docs = [id_to_doc_map[id] for id in retrieved_ids if id in id_to_doc_map]
    
    # 2. 重排序
    reranked_docs, reranked_distances = rerank_documents(
        query, retrieved_docs, distances, embedding_model, top_k
    )
    
    return reranked_docs, reranked_distances

def query_expansion(query: str, embedding_model, chroma_client) -> List[str]:
    """
    查询扩展：生成相关的查询变体
    
    Args:
        query: 原始查询
        embedding_model: 嵌入模型
        chroma_client: ChromaDB 客户端
    
    Returns:
        扩展的查询列表
    """
    expanded_queries = [query]
    
    # 简单的查询扩展策略：
    # 1. 提取关键词
    keywords = extract_keywords(query)
    
    # 2. 生成关键词组合查询
    if len(keywords) >= 2:
        # 选择前两个关键词生成组合查询
        expanded_queries.append(f"{keywords[0]} {keywords[1]}")
        
    # 3. 生成单关键词查询
    for keyword in keywords[:3]:  # 最多3个关键词
        if keyword not in expanded_queries[0]:
            expanded_queries.append(keyword)
    
    return expanded_queries

def remove_duplicate_documents(documents: List[Dict], threshold: float = 0.9) -> List[Dict]:
    """
    移除重复的文档
    
    Args:
        documents: 文档列表
        threshold: 相似度阈值，超过此值视为重复
    
    Returns:
        去重后的文档列表
    """
    if not documents:
        return []
    
    unique_docs = []
    seen_contents = set()
    
    for doc in documents:
        content = doc.get('content', '')
        # 简单的内容哈希去重
        content_hash = hash(content[:500])  # 使用前500字符的哈希
        
        if content_hash not in seen_contents:
            seen_contents.add(content_hash)
            unique_docs.append(doc)
    
    return unique_docs

def multi_hop_retrieval(query: str, chroma_client, embedding_model, 
                        max_hops: int = 2, top_k: int = 3) -> List[Dict]:
    """
    多跳检索：通过迭代检索获取更全面的信息
    
    Args:
        query: 初始查询
        chroma_client: ChromaDB 客户端
        embedding_model: 嵌入模型
        max_hops: 最大跳数
        top_k: 每跳检索的文档数
    
    Returns:
        检索到的文档列表
    """
    all_docs = []
    current_query = query
    
    for hop in range(max_hops):
        # 检索当前查询
        docs, _ = hybrid_search(current_query, chroma_client, embedding_model, top_k)
        
        if not docs:
            break
        
        # 添加到结果集
        all_docs.extend(docs)
        
        # 如果不是最后一跳，生成下一个查询
        if hop < max_hops - 1 and docs:
            # 从检索到的文档中提取关键词作为下一个查询
            top_doc = docs[0]
            keywords = extract_keywords(top_doc.get('content', ''), top_n=3)
            if keywords:
                current_query = f"{query} {keywords[0]}"
    
    # 去重
    return remove_duplicate_documents(all_docs)[:top_k * max_hops]