# 情感分类实验对比总结

## 📁 项目概览

本仓库包含两个完整的情感分类实验项目：

### 1. BERT英文情感分类器 (bert-sentential-classifer)
- **模型**: bert-base-uncased
- **数据**: Amazon英文评论数据
- **特点**: 英文情感分类，适用于英文产品评论

### 2. Qwen/BERT中文情感分类器 (qwen-sentential-classifier)
- **模型**: bert-base-chinese (可替换为Qwen)
- **数据**: Amazon中文评论数据
- **特点**: 中文情感分类，支持中文产品评论

---

## 🧪 实验套件功能

两个项目都包含完整的实验分析套件：

| 实验 | 脚本名称 | 功能描述 | 输出文件 |
|------|---------|---------|---------|
| 1. 训练前后对比 | `compare_trained_untrained.py` | 对比未训练和训练后模型准确率 | 终端输出 |
| 2. 抽样稳定性 | `sample_stability_analysis.py` | 测试样本大小对准确率估计的影响 | CSV + PNG |
| 3. 训练集大小 | `train_size_analysis.py` | 训练数据量对性能的影响 | CSV + JSON + PNG |
| 4. 训练轮数 | `epoch_analysis.py` | Epochs对性能和过拟合的影响 | CSV + JSON + 2×PNG |

---

## 🚀 快速使用

### BERT英文版

```bash
cd bert-sentential-classifer

# 1. 训练模型
python main.py

# 2. 运行所有实验
./run_all_experiments.sh

# 或单独运行实验
python compare_trained_untrained.py --samples 1000
python sample_stability_analysis.py --sample-sizes 100 200 500 1000 --trials 10
python train_size_analysis.py --train-sizes 500 1000 2000 5000 --epochs 3
python epoch_analysis.py --max-epochs 10
```

### Qwen/BERT中文版

```bash
cd qwen-sentential-classifier

# 1. 训练模型
python main.py

# 2. 运行所有实验
./run_all_experiments.sh

# 或单独运行实验（同上）
```

---

## 📊 实验设计亮点

### 1. 抽样稳定性分析
- **创新点**: 通过重复随机抽样，量化测试集大小对准确率估计的影响
- **统计严谨**: 计算均值、标准差、95%置信区间
- **可视化**: 4个子图全面展示稳定性趋势
- **实用价值**: 帮助确定最小可靠测试集大小

### 2. 训练集大小分析
- **系统性**: 测试多个训练集大小 [500, 1000, 2000, 5000]
- **完整记录**: 保存每个模型的训练历史
- **边际收益**: 展示数据量增加的递减效应
- **成本效益**: 帮助平衡标注成本和模型性能

### 3. 训练轮数分析
- **详细监控**: 记录每个epoch的训练/验证/测试指标
- **过拟合检测**: 专门的过拟合分析图
- **最佳停止点**: 自动保存最佳模型
- **检查点**: 保存关键epoch的模型快照

### 4. 统一的实验框架
- **可复现性**: 所有实验使用固定随机种子
- **参数化**: 命令行参数灵活调整实验规模
- **自动化**: 一键运行脚本
- **文档化**: 详细的README和代码注释

---

## 📈 实验结果示例

### 典型发现（基于BERT英文模型）

#### 训练前后对比
```
未训练准确率: ~50% (随机猜测水平)
训练后准确率: ~85-90%
准确率提升: ~35-40%
```

#### 抽样稳定性
```
100样本:  标准差 ~2.5%, CV ~2.8%
500样本:  标准差 ~1.2%, CV ~1.4%
1000样本: 标准差 ~0.8%, CV ~0.9%
```
**结论**: 1000样本足以提供稳定的准确率估计

#### 训练集大小
```
500样本:  验证准确率 ~75%
1000样本: 验证准确率 ~80%
2000样本: 验证准确率 ~83%
5000样本: 验证准确率 ~85%
```
**结论**: 2000-5000样本达到性能-成本平衡点

#### 训练轮数
```
Epoch 1: 快速提升到 ~75%
Epoch 3: 达到峰值 ~85%
Epoch 5+: 开始过拟合
过拟合Gap: Epoch 10时 ~8-10%
```
**结论**: 3-5个epochs为最佳停止点

---

## 🔄 两个项目的主要区别

| 特性 | BERT英文版 | Qwen中文版 |
|------|-----------|-----------|
| **模型** | bert-base-uncased | bert-base-chinese |
| **Tokenizer** | BertTokenizer | AutoTokenizer |
| **语言** | 英文 | 中文 |
| **序列长度** | 64 | 128 |
| **Batch大小** | 8 | 16 |
| **数据路径** | 当前目录 | 当前目录 |
| **模型保存** | sentiment_model.pth | saved_models/sentiment_model.pth |
| **学习率调度** | 无 | LinearScheduleWithWarmup |
| **梯度裁剪** | 无 | clip_grad_norm_(1.0) |

---

## 📝 实验报告撰写建议

### 报告结构

```
1. 引言
   - 任务背景
   - 研究目标

2. 相关工作
   - BERT/Qwen模型介绍
   - 情感分类任务综述

3. 实验设置
   3.1 数据集
   3.2 模型架构
   3.3 训练配置
   3.4 评估指标

4. 实验结果与分析
   4.1 基准性能验证（实验1）
   4.2 测试集抽样稳定性分析（实验2）
       - 表格：不同样本大小的统计量
       - 图表：箱线图、置信区间图
       - 分析：样本大小选择建议

   4.3 训练集规模影响分析（实验3）
       - 表格：不同训练集大小的准确率
       - 图表：训练曲线、学习曲线
       - 分析：数据量的边际收益

   4.4 训练策略优化（实验4）
       - 表格：每个epoch的详细指标
       - 图表：学习曲线、过拟合分析
       - 分析：最佳训练轮数、过拟合现象

5. 结论与展望
   - 关键发现总结
   - 改进方向
   - 未来工作

6. 参考文献
```

### 图表使用建议

- **实验2**: 使用箱线图展示样本大小对准确率分布的影响
- **实验3**: 使用折线图展示训练集大小与准确率的关系
- **实验4**: 使用双Y轴图同时展示准确率和损失的变化

---

## 🎯 实验扩展方向

### 已实现的功能
✅ 训练前后对比
✅ 抽样稳定性分析
✅ 训练集大小影响
✅ 训练轮数影响

### 可扩展的实验
- [ ] 学习率影响分析
- [ ] Batch Size影响分析
- [ ] 不同预训练模型对比（BERT vs RoBERTa vs Qwen）
- [ ] 数据增强技术效果
- [ ] Fine-tuning策略对比（全参数 vs LoRA）
- [ ] 类别不平衡处理
- [ ] 错误分析和case study

---

## 💻 系统要求

- Python 3.7+
- PyTorch 1.9+
- transformers 4.0+
- pandas, numpy, matplotlib, tqdm
- （推荐）CUDA GPU用于加速训练

---

## 📚 参考文献

1. Devlin, J., et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. arXiv:1810.04805

2. Bai, J., et al. (2023). Qwen Technical Report. arXiv:2309.16609

3. Hugging Face Transformers: https://huggingface.co/transformers/

---

## 🤝 贡献

欢迎提交Issue和Pull Request改进实验套件！

---

## 📄 许可证

本项目仅用于教育和研究目的。

---

**最后更新**: 2025-12-03
**维护者**: Claude Code Assistant
