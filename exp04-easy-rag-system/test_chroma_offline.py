#!/usr/bin/env python3
"""
离线模式 ChromaDB 测试脚本
验证 ChromaDB 可以在完全离线的情况下创建 collection 和索引数据
"""

import os
import sys

# 设置离线环境变量
os.environ['HF_HOME'] = './hf_cache'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'

import chromadb
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL_NAME, EMBEDDING_DIM, COLLECTION_NAME

class DummyEmbeddingFunction(chromadb.utils.embedding_functions.EmbeddingFunction):
    """虚拟嵌入函数，防止 ChromaDB 自动下载模型"""
    def __call__(self, input):
        if isinstance(input, str):
            input = [input]
        return [[0.0] * EMBEDDING_DIM for _ in input]

def test_offline_chroma():
    print("=" * 60)
    print("🧪 离线模式 ChromaDB 测试")
    print("=" * 60)
    
    # 步骤1：初始化 ChromaDB 客户端
    print("\n[步骤 1] 初始化 ChromaDB 客户端...")
    try:
        persist_dir = "./chroma_data_offline_test"
        if not os.path.exists(persist_dir):
            os.makedirs(persist_dir, exist_ok=True)
        
        if hasattr(chromadb, "PersistentClient"):
            client = chromadb.PersistentClient(path=persist_dir)
            print(f"✅ 使用 PersistentClient")
        else:
            settings = chromadb.config.Settings(persist_directory=persist_dir)
            client = chromadb.Client(settings)
            print(f"✅ 使用旧版本 Client")
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        return False
    
    # 步骤2：删除旧的 collection（如果存在）
    print("\n[步骤 2] 清理旧的 collection...")
    try:
        client.delete_collection(name=COLLECTION_NAME)
        print(f"✅ 已删除旧的 collection")
    except:
        print(f"✅ 没有旧的 collection 需要删除（这是正常的）")
    
    # 步骤3：使用虚拟嵌入函数创建 collection
    print("\n[步骤 3] 创建 collection (使用虚拟嵌入函数)...")
    try:
        dummy_embedding_fn = DummyEmbeddingFunction()
        collection = client.create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
            embedding_function=dummy_embedding_fn
        )
        print(f"✅ Collection '{COLLECTION_NAME}' 创建成功！")
        print(f"   当前文档数: {collection.count()}")
    except Exception as e:
        print(f"❌ 创建 collection 失败: {e}")
        return False
    
    # 步骤4：加载嵌入模型
    print("\n[步骤 4] 加载嵌入模型...")
    try:
        cache_path = os.path.abspath('./hf_cache')
        print(f"   使用缓存路径: {cache_path}")
        model = SentenceTransformer(EMBEDDING_MODEL_NAME, cache_folder=cache_path, device='cpu')
        print(f"✅ 嵌入模型加载成功")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return False
    
    # 步骤5：测试数据索引
    print("\n[步骤 5] 测试数据索引...")
    try:
        test_docs = [
            "这是关于中医诊断的文章",
            "中医药物治疗方法研究",
            "针灸和拔罐疗法的临床应用"
        ]
        
        # 生成嵌入
        embeddings = model.encode(test_docs, show_progress_bar=False)
        print(f"   生成了 {len(embeddings)} 个嵌入向量")
        print(f"   向量维度: {len(embeddings[0])}")
        
        # 添加到 collection
        collection.add(
            ids=[str(i) for i in range(len(test_docs))],
            embeddings=embeddings.tolist(),
            documents=test_docs,
            metadatas=[{"source": f"test_{i}"} for i in range(len(test_docs))]
        )
        print(f"✅ 成功索引 {len(test_docs)} 个文档")
        print(f"   collection 中的文档总数: {collection.count()}")
    except Exception as e:
        print(f"❌ 索引失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 步骤6：测试搜索
    print("\n[步骤 6] 测试搜索功能...")
    try:
        query = "中医治疗"
        query_embedding = model.encode([query])[0]
        
        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=3
        )
        
        print(f"   查询: '{query}'")
        print(f"   找到 {len(results['ids'][0])} 个结果:")
        for i, doc_id in enumerate(results['ids'][0]):
            distance = results['distances'][0][i] if results['distances'] else 'N/A'
            doc_text = results['documents'][0][i] if results['documents'] else 'N/A'
            print(f"     [{i+1}] ID: {doc_id}, 距离: {distance:.4f}")
            print(f"         内容: {doc_text[:50]}...")
        
        print(f"✅ 搜索功能正常")
    except Exception as e:
        print(f"❌ 搜索失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 60)
    print("✅ 所有测试通过！离线模式工作正常")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = test_offline_chroma()
    sys.exit(0 if success else 1)
