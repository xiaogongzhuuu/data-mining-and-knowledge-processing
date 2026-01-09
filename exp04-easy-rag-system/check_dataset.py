import json

# 读取数据集
with open('./data/processed_data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 统计数据集信息
print("="*60)
print("数据集统计信息")
print("="*60)
print(f"\n总文档数: {len(data)}")

# 统计每个作者的文档数量
authors = {}
for doc in data:
    source = doc.get('source_file', 'unknown')
    authors[source] = authors.get(source, 0) + 1

print(f"\n数据来源分布:")
print("-"*60)
for author, count in sorted(authors.items()):
    print(f"  {author}: {count} 篇")

print(f"\n数据集特征:")
print(f"  - 数据文件: ./data/processed_data.json")
print(f"  - 原始来源: ./sources/ 目录")
print(f"  - 字段结构: id, title, abstract, source_file, chunk_index")
print(f"  - 内容类型: 中医专家学术经验集萃")

print("\n" + "="*60)