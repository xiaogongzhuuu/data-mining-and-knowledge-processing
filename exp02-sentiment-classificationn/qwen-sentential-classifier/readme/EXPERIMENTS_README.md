# Qwen情感分类实验分析套件

## 📋 实验概述

本套件包含完整的情感分类模型实验分析工具，基于BERT中文模型（可替换为Qwen模型）：

1. **训练前后对比** - 对比未训练和已训练模型的准确率差异
2. **测试集抽样稳定性分析** - 研究测试集样本大小对准确率估计的影响
3. **训练集大小影响分析** - 分析训练数据量对模型性能的影响
4. **训练轮数影响分析** - 研究训练epochs对模型性能和过拟合的影响

---

## 📁 项目结构

```
qwen-sentential-classifier/
├── config.py                          # 配置文件
├── model.py                           # 模型定义
├── dataset.py                         # 数据集类
├── load_data.py                       # 数据加载器
├── main.py                            # 训练主程序
├── compare_trained_untrained.py       # 实验1: 训练前后对比
├── sample_stability_analysis.py       # 实验2: 抽样稳定性分析
├── train_size_analysis.py             # 实验3: 训练集大小影响
├── epoch_analysis.py                  # 实验4: 训练轮数影响
├── run_all_experiments.sh             # 一键运行所有实验
├── EXPERIMENTS_README.md              # 本文档
├── train.csv                          # 训练数据
├── dev.csv                            # 验证数据
└── test.csv                           # 测试数据
```

---

## 🚀 快速开始

### 1. 环境准备

```bash
pip install -r requirements.txt
```

### 2. 训练模型

首先需要训练一个基础模型：

```bash
python main.py
```

这会在 `saved_models/` 目录下生成 `sentiment_model.pth`。

### 3. 运行实验

#### 方式1: 一键运行所有实验

```bash
chmod +x run_all_experiments.sh
./run_all_experiments.sh
```

#### 方式2: 单独运行实验

见下方各实验详解。

---

## 📊 实验详解

### 实验1: 训练前后模型对比

**脚本**: `compare_trained_untrained.py`

**功能**: 对比未训练模型（随机初始化分类层）和训练后模型的准确率。

**使用方法**:
```bash
# 默认评估1000个样本
python compare_trained_untrained.py

# 自定义样本数
python compare_trained_untrained.py --samples 500

# 评估完整测试集
python compare_trained_untrained.py --full-test
```

**输出示例**:
```
测试样本数:       1000
未训练准确率:     49.8%  (498/1000)
训练后准确率:     88.5%  (885/1000)
准确率提升:       38.7%
相对提升:         77.7%
```

---

### 实验2: 测试集抽样稳定性分析

**脚本**: `sample_stability_analysis.py`

**功能**: 研究不同测试集样本大小对准确率估计稳定性的影响。

**研究问题**:
- 多大的测试集样本能够稳定估计模型准确率？
- 样本大小对准确率方差的影响？

**使用方法**:
```bash
# 默认配置: sample_sizes=[100,200,500,1000], trials=10
python sample_stability_analysis.py

# 自定义参数
python sample_stability_analysis.py --sample-sizes 100 200 500 1000 2000 --trials 20
```

**参数说明**:
- `--sample-sizes`: 测试的样本大小列表
- `--trials`: 每个样本大小重复抽样的次数

**输出文件**:
- `sample_stability_results.csv`: 详细结果数据
- `sample_stability_plot.png`: 可视化图表（4个子图）

**预期结果**:
- 样本越大，准确率估计的标准差越小
- 1000样本通常能提供稳定的准确率估计

---

### 实验3: 训练集大小影响分析

**脚本**: `train_size_analysis.py`

**功能**: 使用不同大小的训练集训练模型，观察训练数据量对模型性能的影响。

**研究问题**:
- 训练数据量和模型性能的关系？
- 达到良好性能需要多少训练数据？
- 是否存在边际收益递减？

**使用方法**:
```bash
# 默认配置: train_sizes=[500,1000,2000,5000], epochs=3
python train_size_analysis.py

# 自定义参数
python train_size_analysis.py --train-sizes 500 1000 2000 5000 10000 --epochs 5 --val-size 1000
```

**参数说明**:
- `--train-sizes`: 训练集大小列表
- `--epochs`: 每个模型训练的轮数
- `--val-size`: 验证集大小

**输出文件**:
- `train_size_results.csv`: 汇总结果
- `train_size_history.json`: 详细训练历史
- `train_size_analysis.png`: 可视化图表
- `model_trainsize_*.pth`: 各训练集大小对应的模型文件

**⚠️ 注意**: 此实验需要重新训练多个模型，耗时较长。

---

### 实验4: 训练轮数影响分析

**脚本**: `epoch_analysis.py`

**功能**: 训练模型多个epochs，记录每个epoch的性能，分析学习曲线和过拟合现象。

**研究问题**:
- 训练轮数对模型性能的影响？
- 何时开始出现过拟合？
- 最佳停止点在哪里？

**使用方法**:
```bash
# 默认配置: max_epochs=10
python epoch_analysis.py

# 自定义参数
python epoch_analysis.py --max-epochs 15 --train-size 5000 --val-size 1000 --test-size 1000 --checkpoint-epochs 1 3 5 10 15
```

**参数说明**:
- `--max-epochs`: 最大训练轮数
- `--train-size`: 训练集大小
- `--val-size`: 验证集大小
- `--test-size`: 测试集大小
- `--checkpoint-epochs`: 保存模型检查点的epoch列表

**输出文件**:
- `epoch_analysis_results.csv`: 每个epoch的详细指标
- `epoch_analysis_history.json`: 完整训练历史
- `epoch_analysis_learning_curves.png`: 学习曲线（准确率和损失）
- `epoch_analysis_overfitting.png`: 过拟合分析图
- `best_epoch_model.pth`: 验证集上表现最佳的模型
- `model_epoch_*.pth`: 指定epoch的检查点

**关键指标**:
- **Overfitting Gap**: Train Acc - Val Acc
  - < 0.05: 过拟合最小
  - 0.05 - 0.10: 中度过拟合
  - \> 0.10: 严重过拟合

---

## 📈 实验结果解读

### 样本稳定性分析

**观察指标**:
- **标准差**: 越小表示估计越稳定
- **变异系数 (CV)**: std/mean，归一化的变异度量
- **95% 置信区间**: 真实准确率的估计范围

**典型发现**:
- 100样本: 标准差约2-3%
- 500样本: 标准差约1-1.5%
- 1000样本: 标准差约0.5-1%

### 训练集大小分析

**观察模式**:
1. **线性增长期**: 数据量小时，增加数据显著提升性能
2. **对数增长期**: 性能提升放缓
3. **饱和期**: 更多数据带来的收益很小

### Epoch分析

**健康学习曲线特征**:
- 训练和验证曲线同步上升
- 验证曲线在某点后趋于平稳
- 训练-验证gap保持较小

**过拟合警示信号**:
- 训练准确率持续上升，验证准确率下降
- 训练-验证gap快速增大
- 验证损失开始上升

---

## 🔧 配置说明

修改 [config.py](config.py) 中的参数：

```python
class Config:
    model_name = "bert-base-chinese"  # 可改为 "Qwen/Qwen-xxx"
    max_seq_length = 128
    num_classes = 2
    batch_size = 16
    learning_rate = 2e-5
    num_epochs = 5
```

根据硬件条件调整：
- `batch_size`: 显存不足时减小
- `max_seq_length`: 减小可节省内存
- `num_epochs`: 增大可能提升性能但需更多时间

---

## 📝 实验报告建议结构

### 1. 实验目的
- 评估模型在中文情感分类任务上的表现
- 分析各种因素对模型性能的影响

### 2. 实验设置
- 数据集描述（Amazon中文评论数据）
- 模型架构和参数
- 实验环境

### 3. 实验结果

#### 3.1 训练效果验证
- 展示训练前后对比结果
- 证明训练有效性

#### 3.2 抽样稳定性分析
- 表格和图表展示不同样本大小的准确率分布
- 分析：样本大小对准确率估计稳定性的影响

#### 3.3 训练集规模影响
- 绘制训练集大小 vs 准确率曲线
- 分析：数据量的边际收益

#### 3.4 训练策略分析
- 展示学习曲线
- 分析过拟合现象
- 建议最佳训练轮数

### 4. 结论与建议
- 总结关键发现
- 提出改进方向

---

## 💡 使用建议

1. **首次运行**: 先运行实验1和2，快速且不需要重新训练
2. **深入分析**: 再运行实验3和4，需要更多时间
3. **撰写报告**: 使用生成的CSV和PNG文件

---

## 🐛 常见问题

**Q: 显存不足怎么办？**

A: 在 `config.py` 中减小 `batch_size` 和 `max_seq_length`

**Q: 实验时间太长？**

A:
- 减少 `--trials` 参数
- 减少 `--train-sizes` 或 `--max-epochs`
- 使用更小的 `--train-size` 和 `--val-size`

**Q: 如何确保结果可复现？**

A: 所有脚本都设置了随机种子（seed=42），结果应该可复现

**Q: 如何使用Qwen模型？**

A: 修改 `config.py` 中的 `model_name` 为 Qwen 模型名称，如：
```python
model_name = "Qwen/Qwen-7B"  # 或其他Qwen模型
```

---

## 📚 参考资料

- BERT: [Devlin et al., 2018](https://arxiv.org/abs/1810.04805)
- Qwen: [Alibaba Cloud](https://github.com/QwenLM/Qwen)
- Hugging Face Transformers: https://huggingface.co/transformers/

---

**创建日期**: 2025-12-03
**作者**: Claude Code Assistant
