"""
测试配置脚本：验证数据加载和均衡采样
"""

import config
from data_loader import create_data_loaders

print("="*80)
print("TextCNN 配置测试")
print("="*80)

print("\n📋 当前配置:")
print(f"  MAX_TRAIN_SAMPLES: {config.MAX_TRAIN_SAMPLES}")
print(f"  BALANCE_TRAIN_DATA: {config.BALANCE_TRAIN_DATA}")
print(f"  BATCH_SIZE: {config.BATCH_SIZE}")
print(f"  NUM_EPOCHS: {config.NUM_EPOCHS}")
print(f"  EMBEDDING_DIM: {config.EMBEDDING_DIM}")
print(f"  NUM_FILTERS: {config.NUM_FILTERS}")
print(f"  FILTER_SIZES: {config.FILTER_SIZES}")
print(f"  MAX_SEQ_LENGTH: {config.MAX_SEQ_LENGTH}")

print("\n" + "="*80)
print("测试数据加载...")
print("="*80)

try:
    train_loader, dev_loader, test_loader, vocab = create_data_loaders()
    
    print("\n✅ 数据加载成功！")
    print(f"\n📊 数据集信息:")
    print(f"  训练集: {len(train_loader.dataset)} 样本")
    print(f"  验证集: {len(dev_loader.dataset)} 样本")
    print(f"  测试集: {len(test_loader.dataset)} 样本")
    print(f"  词表大小: {len(vocab)}")
    
    # 检查样本分布
    from collections import Counter
    train_labels = [train_loader.dataset[i]['labels'].item() for i in range(len(train_loader.dataset))]
    train_dist = Counter(train_labels)
    
    print(f"\n⚖️  训练集标签分布:")
    print(f"  负面 (0): {train_dist[0]} ({train_dist[0]/len(train_labels)*100:.1f}%)")
    print(f"  正面 (1): {train_dist[1]} ({train_dist[1]/len(train_labels)*100:.1f}%)")
    
    if abs(train_dist[0] - train_dist[1]) < 100:
        print(f"  ✅ 样本分布均衡！")
    else:
        print(f"  ⚠️  样本分布不均衡")
    
    print("\n" + "="*80)
    print("测试完成！准备开始训练。")
    print("="*80)
    print("\n运行以下命令开始训练:")
    print("  python train.py")
    
except Exception as e:
    print(f"\n❌ 错误: {e}")
    import traceback
    traceback.print_exc()


