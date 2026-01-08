"""
测试嵌入向量和距离计算
"""

import os
import sys
import numpy as np

os.environ['HF_HOME'] = './hf_cache'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'

from sentence_transformers import SentenceTransformer
import chromadb

print("="*60)
print("嵌入向量和距离计算测试")
print("="*60)

# 1. 加载模型
print("\n1. 加载模型...")
model_path = "./hf_cache/hub/models--moka-ai--m3e-base/snapshots/764b537a0e50e5c7d64db883f2d2e051cbe3c64c"
model = SentenceTransformer(model_path, device='cpu')
print(f"模型加载成功，维度: {model.get_sentence_embedding_dimension()}")

# 2. 测试嵌入
print("\n2. 测试嵌入...")
test_texts = [
    "感冒了咋办",
    "吴银根的学术思想",
    "如何治疗哮喘"
]

embeddings = model.encode(test_texts)
print(f"生成了 {len(embeddings)} 个嵌入向量")
print(f"每个向量维度: {embeddings[0].shape}")
print(f"向量值范围: [{embeddings.min():.4f}, {embeddings.max():.4f}]")
print(f"向量均值: {embeddings.mean():.4f}")
print(f"向量标准差: {embeddings.std():.4f}")

# 3. 计算距离
print("\n3. 计算余弦距离...")
for i in range(len(test_texts)):
    for j in range(i+1, len(test_texts)):
        # 余弦相似度
        dot_product = np.dot(embeddings[i], embeddings[j])
        norm_i = np.linalg.norm(embeddings[i])
        norm_j = np.linalg.norm(embeddings[j])
        cosine_similarity = dot_product / (norm_i * norm_j)

        # 余弦距离 = 1 - 余弦相似度
        cosine_distance = 1 - cosine_similarity

        print(f"\n'{test_texts[i]}' vs '{test_texts[j]}':")
        print(f"  余弦相似度: {cosine_similarity:.4f}")
        print(f"  余弦距离: {cosine_distance:.4f}")

# 4. 检查ChromaDB中的嵌入
print("\n4. 检查ChromaDB中的嵌入...")
client = chromadb.PersistentClient(path=".")
collection = client.get_collection(name="medical_rag_lite")

# 获取一个文档的嵌入
results = collection.get(limit=1, include=['embeddings', 'documents'])
if results['embeddings'] is not None and len(results['embeddings']) > 0:
    db_embedding = np.array(results['embeddings'][0])
    print(f"数据库中的嵌入维度: {db_embedding.shape}")
    print(f"数据库嵌入值范围: [{db_embedding.min():.4f}, {db_embedding.max():.4f}]")
    print(f"数据库嵌入均值: {db_embedding.mean():.4f}")
    print(f"数据库嵌入标准差: {db_embedding.std():.4f}")

    # 计算查询与数据库文档的距离
    print("\n5. 计算查询与数据库文档的距离...")
    for i, text in enumerate(test_texts):
        query_embedding = embeddings[i]

        # 余弦距离
        dot_product = np.dot(query_embedding, db_embedding)
        norm_query = np.linalg.norm(query_embedding)
        norm_db = np.linalg.norm(db_embedding)
        cosine_similarity = dot_product / (norm_query * norm_db)
        cosine_distance = 1 - cosine_similarity

        print(f"\n查询: '{text}'")
        print(f"  余弦相似度: {cosine_similarity:.4f}")
        print(f"  余弦距离: {cosine_distance:.4f}")

print("\n" + "="*60)
print("测试完成")
print("="*60)