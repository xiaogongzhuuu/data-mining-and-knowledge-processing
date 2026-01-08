"""
数据分块优化模块
提供智能分块、重叠分块等功能，提升检索效果
"""

import re
from typing import List, Dict

def smart_chunk_text(text: str, max_length: int = 1000, overlap: int = 100) -> List[str]:
    """
    智能文本分块：在语义边界（句子、段落）处切分
    
    Args:
        text: 输入文本
        max_length: 每个块的最大长度
        overlap: 块之间的重叠长度
    
    Returns:
        分块后的文本列表
    """
    if len(text) <= max_length:
        return [text]
    
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        # 确定块的结束位置
        end = min(start + max_length, text_len)
        
        # 如果不是最后一部分，尝试在句子边界切分
        if end < text_len:
            # 寻找最近的句号、问号、感叹号
            for sep in ['。', '！', '？', '\n\n', '\n']:
                last_sep = text.rfind(sep, start, end)
                if last_sep != -1 and last_sep > start + max_length * 0.5:
                    end = last_sep + 1
                    break
        
        # 提取块
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        # 移动到下一个块的开始位置（考虑重叠）
        start = end - overlap
        if start < 0:
            start = 0
    
    return chunks

def optimize_data_chunks(data: List[Dict], chunk_size: int = 1000, 
                        overlap: int = 100) -> List[Dict]:
    """
    优化数据分块策略
    
    Args:
        data: 原始数据列表
        chunk_size: 每个块的大小
        overlap: 块之间的重叠大小
    
    Returns:
        优化后的数据列表
    """
    optimized_data = []
    
    for doc in data:
        title = doc.get('title', '')
        abstract = doc.get('abstract', '')
        source_file = doc.get('source_file', '')
        
        # 合并标题和摘要
        full_text = f"{title}\n\n{abstract}" if abstract else title
        
        # 智能分块
        chunks = smart_chunk_text(full_text, chunk_size, overlap)
        
        # 为每个块创建单独的文档
        for i, chunk in enumerate(chunks):
            optimized_data.append({
                'title': title,
                'abstract': chunk,
                'content': chunk,
                'source_file': source_file,
                'chunk_index': i,
                'total_chunks': len(chunks)
            })
    
    return optimized_data

def enhance_document_metadata(doc: Dict) -> Dict:
    """
    增强文档元数据，添加更多检索线索
    
    Args:
        doc: 原始文档
    
    Returns:
        增强后的文档
    """
    enhanced = doc.copy()
    
    # 提取关键词
    content = doc.get('content', '')
    keywords = extract_document_keywords(content)
    enhanced['keywords'] = keywords
    
    # 计算文档长度
    enhanced['content_length'] = len(content)
    
    # 添加文档类型标记
    if '学术思想' in content:
        enhanced['doc_type'] = '学术思想'
    elif '临床经验' in content:
        enhanced['doc_type'] = '临床经验'
    elif '治疗' in content or '方药' in content:
        enhanced['doc_type'] = '治疗方案'
    else:
        enhanced['doc_type'] = '综合'
    
    return enhanced

def extract_document_keywords(content: str, top_n: int = 10) -> List[str]:
    """
    从文档中提取关键词
    
    Args:
        content: 文档内容
        top_n: 返回的关键词数量
    
    Returns:
        关键词列表
    """
    # 常见中医关键词
    tcm_keywords = {
        '气血', '阴阳', '脏腑', '经络', '辨证', '施治',
        '寒热', '虚实', '表里', '升降', '出入', '补泻',
        '肺', '肾', '肝', '心', '脾', '胃', '胆', '大肠', '小肠',
        '咳嗽', '哮喘', '水肿', '痰饮', '瘀血', '气滞',
        '方药', '汤剂', '丸剂', '散剂', '针灸', '推拿'
    }
    
    keywords = []
    content_lower = content.lower()
    
    # 查找中医关键词
    for keyword in tcm_keywords:
        if keyword in content:
            keywords.append(keyword)
            if len(keywords) >= top_n:
                break
    
    return keywords
