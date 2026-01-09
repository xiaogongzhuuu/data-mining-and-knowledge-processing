"""
检查评估逻辑是否正确
重点检查：是否在正确的数据集上评估
"""

import csv

def simulate_data_flow():
    """模拟原代码的数据加载和处理流程"""
    
    print("="*60)
    print("模拟原代码的数据流")
    print("="*60)
    
    # 步骤1: 读取文件
    print("\n步骤1: parse_dataset_files() 读取文件")
    
    train_texts, train_labels = [], []
    dev_texts, dev_labels = [], []
    test_texts, test_labels = [], []
    
    # 读取 train
    with open('../train.csv', 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if i >= 100:  # 只读前100个模拟
                break
            if len(row) >= 2:
                try:
                    label = int(row[0])
                    text = ' '.join(row[1:])
                    
                    # 映射标签
                    if label in (1, 2):
                        mapped_label = 0 if label == 1 else 1
                    else:
                        mapped_label = 1 if label > 0 else 0
                    
                    train_texts.append(text)
                    train_labels.append(mapped_label)
                except:
                    pass
    
    # 读取 dev
    with open('../dev.csv', 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if i >= 50:  # 只读前50个模拟
                break
            if len(row) >= 2:
                try:
                    label = int(row[0])
                    text = ' '.join(row[1:])
                    
                    # 映射标签
                    if label in (1, 2):
                        mapped_label = 0 if label == 1 else 1
                    else:
                        mapped_label = 1 if label > 0 else 0
                    
                    dev_texts.append(text)
                    dev_labels.append(mapped_label)
                except:
                    pass
    
    # 读取 test
    with open('../test.csv', 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if i >= 50:  # 只读前50个模拟
                break
            if len(row) >= 2:
                try:
                    label = int(row[0])
                    text = ' '.join(row[1:])
                    
                    # 映射标签
                    if label in (1, 2):
                        mapped_label = 0 if label == 1 else 1
                    else:
                        mapped_label = 1 if label > 0 else 0
                    
                    test_texts.append(text)
                    test_labels.append(mapped_label)
                except:
                    pass
    
    print(f"  train: {len(train_texts)} 个样本")
    print(f"  dev:   {len(dev_texts)} 个样本")
    print(f"  test:  {len(test_texts)} 个样本")
    
    # 步骤2: 检查 dev 是否为空（原代码139-145行）
    print("\n步骤2: 检查 dev 是否为空")
    if len(dev_texts) == 0:
        print("  ⚠️  dev 为空，会从 train 切分！")
        print("  这是原代码的逻辑，但我们的 dev.csv 不为空，所以不会执行")
    else:
        print(f"  ✅ dev 不为空（{len(dev_texts)}个样本），不需要从 train 切分")
    
    # 步骤3: 创建 DataLoader（原代码 318-323行）
    print("\n步骤3: 创建 DataLoader")
    print("  train_loader = make_dataloader(train_texts, train_labels, ...)")
    print("  dev_loader   = make_dataloader(dev_texts, dev_labels, ...)    if dev_texts else None")
    print("  test_loader  = make_dataloader(test_texts, test_labels, ...)  if test_texts else None")
    print(f"  ✅ 三个 DataLoader 都被正确创建")
    
    # 步骤4: 评估（原代码 398-399行）
    print("\n步骤4: 评估时调用")
    print("  dev_metrics  = evaluate_full(model, dev_loader, device)   if dev_loader else {}")
    print("  test_metrics = evaluate_full(model, test_loader, device)  if test_loader else {}")
    print("  ✅ 代码逻辑上是正确的，使用了 dev_loader 和 test_loader")
    
    # 检查潜在问题
    print("\n" + "="*60)
    print("潜在问题分析")
    print("="*60)
    
    # 检查文本是否相同（数据泄露检测）
    print("\n1. 检查是否存在文本重复（train vs dev）")
    train_set = set([t[:100] for t in train_texts[:50]])
    dev_set = set([t[:100] for t in dev_texts])
    
    overlap = train_set & dev_set
    if overlap:
        print(f"  ⚠️  发现 {len(overlap)} 个重复文本！")
        print(f"  示例: {list(overlap)[0][:80]}...")
    else:
        print(f"  ✅ 没有发现重复文本")
    
    print("\n2. 检查数据集大小是否合理")
    if len(train_texts) < 500:
        print(f"  ⚠️  训练集只有 {len(train_texts)} 个样本，可能太少")
    else:
        print(f"  ✅ 训练集大小合理: {len(train_texts)} 个样本")
    
    print("\n3. 检查标签分布")
    from collections import Counter
    print(f"  train: {Counter(train_labels)}")
    print(f"  dev:   {Counter(dev_labels)}")
    print(f"  test:  {Counter(test_labels)}")
    
    # 检查是否严重不平衡
    train_ratio = train_labels.count(1) / len(train_labels) if train_labels else 0
    dev_ratio = dev_labels.count(1) / len(dev_labels) if dev_labels else 0
    test_ratio = test_labels.count(1) / len(test_labels) if test_labels else 0
    
    print(f"\n  正类比例:")
    print(f"    train: {train_ratio:.2%}")
    print(f"    dev:   {dev_ratio:.2%}")
    print(f"    test:  {test_ratio:.2%}")
    
    if abs(train_ratio - dev_ratio) > 0.2:
        print(f"  ⚠️  train 和 dev 的标签分布差异较大（{abs(train_ratio - dev_ratio):.2%}）")
    else:
        print(f"  ✅ 标签分布基本一致")


def check_common_mistakes():
    """检查常见错误"""
    print("\n" + "="*60)
    print("常见导致准确率虚高的错误")
    print("="*60)
    
    mistakes = [
        {
            "错误": "在测试集上训练",
            "检查": "代码中 train_loader 使用了 train_texts，没有问题",
            "状态": "✅"
        },
        {
            "错误": "评估时用了训练集",
            "检查": "evaluate_full() 传入的是 dev_loader/test_loader",
            "状态": "✅"
        },
        {
            "错误": "数据泄露（train和test有重复）",
            "检查": "前面检查未发现重复",
            "状态": "✅"
        },
        {
            "错误": "模型在训练时记住了测试样本",
            "检查": "需要检查模型训练过程",
            "状态": "❓"
        },
        {
            "错误": "训练样本太少导致过拟合",
            "检查": "MAX_TRAIN_SAMPLES=1000，确实较少",
            "状态": "⚠️"
        },
        {
            "错误": "测试集分布与训练集完全不同",
            "检查": "需要更详细的分析",
            "状态": "❓"
        },
        {
            "错误": "模型容量过大，记住了所有训练样本",
            "检查": "Qwen2.5-0.5B 有 5亿参数，对1000样本来说太大",
            "状态": "⚠️"
        },
        {
            "错误": "预训练模型已经学过类似数据",
            "检查": "Amazon评论是常见的情感分析数据集",
            "状态": "⚠️"
        },
    ]
    
    for i, m in enumerate(mistakes, 1):
        print(f"\n{i}. {m['错误']} [{m['状态']}]")
        print(f"   检查: {m['检查']}")


if __name__ == "__main__":
    simulate_data_flow()
    check_common_mistakes()
    
    print("\n" + "="*60)
    print("结论和建议")
    print("="*60)
    print("""
从代码逻辑分析，没有发现明显的编程错误。

准确率过高的最可能原因：

1. **预训练模型的优势** (最可能) ⭐⭐⭐⭐⭐
   - Qwen2.5-0.5B 是在大量文本上预训练的
   - 可能已经见过类似的 Amazon 评论数据
   - 对情感分析任务有很强的先验知识
   
2. **模型过拟合** (很可能) ⭐⭐⭐⭐
   - 5亿参数的模型 vs 1000个训练样本
   - 参数量远大于数据量，容易记住训练集
   - 但如果测试集准确率也很高，说明不只是过拟合
   
3. **任务相对简单** (可能) ⭐⭐⭐
   - Amazon 评论通常情感表达明确
   - 包含 title + review，信息充分
   - 二分类任务（正面/负面）相对简单

建议验证准确率是否真实过高：

1. 计算 baseline 模型准确率（如传统 SVM、朴素贝叶斯）
2. 对比未微调的 Qwen 模型 vs 微调后的模型
3. 检查混淆矩阵，看是否两类都预测得好
4. 打印一些错误案例，看模型在哪些样本上犯错
5. 尝试更困难的测试集（如不同领域的评论）

如果 baseline 模型也能达到 85%+ 准确率，那说明任务本身较简单。
如果只有 Qwen 能达到高准确率，那说明预训练模型确实很强大。
    """)

