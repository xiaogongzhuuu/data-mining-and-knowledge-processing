import json

# 读取数据
with open('./data/processed_data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 统计每个作者的文档数量
authors = {}
for doc in data:
    source = doc.get('source_file', 'unknown')
    authors[source] = authors.get(source, 0) + 1

# 打印统计结果
print("=" * 60)
print("数据源作者分布统计")
print("=" * 60)
print(f"总文档数: {len(data)}")
print(f"作者数量: {len(authors)}")
print("\n各作者文档数量:")
print("-" * 60)

# 按文档数量排序
sorted_authors = sorted(authors.items(), key=lambda x: x[1], reverse=True)
for author, count in sorted_authors:
    print(f"  {author}: {count} 篇")

# 检查检索到的 ID
print("\n" + "=" * 60)
print("检索结果分析")
print("=" * 60)
hit_ids = [328, 331, 337, 329, 327, 13, 336, 323, 146, 348]
print(f"检索到的文档 ID: {hit_ids}")
print("\n这些文档的详细信息:")
for doc_id in hit_ids:
    if doc_id < len(data):
        doc = data[doc_id]
        print(f"  ID {doc_id}: {doc.get('source_file', 'unknown')} - {doc.get('title', 'no title')}")
    else:
        print(f"  ID {doc_id}: 超出范围")

# 统计检索结果中的作者分布
print("\n检索结果中的作者分布:")
result_authors = {}
for doc_id in hit_ids:
    if doc_id < len(data):
        source = data[doc_id].get('source_file', 'unknown')
        result_authors[source] = result_authors.get(source, 0) + 1

for author, count in sorted(result_authors.items(), key=lambda x: x[1], reverse=True):
    print(f"  {author}: {count} 篇")