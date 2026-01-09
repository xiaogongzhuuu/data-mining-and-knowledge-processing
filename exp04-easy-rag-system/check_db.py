"""
检查ChromaDB中的数据
"""

import chromadb

client = chromadb.PersistentClient(path=".")

try:
    collection = client.get_collection(name="medical_rag_lite")
    print(f"Collection found: medical_rag_lite")
    print(f"Entity count: {collection.count()}")

    # 尝试获取一些数据
    print("\n尝试获取前3个文档:")
    results = collection.get(limit=3, include=['embeddings', 'documents', 'metadatas'])
    print(f"Retrieved {len(results['ids'])} documents")

    for i in range(len(results['ids'])):
        print(f"\nDocument {i+1}:")
        print(f"  ID: {results['ids'][i]}")
        print(f"  Title: {results['metadatas'][i].get('title', 'N/A')}")
        if results['embeddings'] is not None and len(results['embeddings']) > 0:
            print(f"  Embedding dimension: {len(results['embeddings'][i])}")
        print(f"  Content preview: {results['documents'][i][:100]}...")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()