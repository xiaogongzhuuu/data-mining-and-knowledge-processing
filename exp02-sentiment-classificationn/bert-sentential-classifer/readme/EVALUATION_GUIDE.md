# BERT情感分类 - 完整评估指南

本指南介绍如何使用新增的评估脚本进行全面的模型评估和对比实验。

## 📋 新增评估功能

### ✅ 已实现的评估指标

1. **综合评估指标**
   - ✅ 准确率 (Accuracy)
   - ✅ 精确率 (Precision)
   - ✅ 召回率 (Recall)
   - ✅ F1分数 (F1-score)
   - ✅ AUC-ROC
   - ✅ 混淆矩阵可视化
   - ✅ ROC曲线

2. **鲁棒性测试**
   - ✅ K折交叉验证

3. **对比实验**
   - ✅ 传统机器学习模型（SVM、朴素贝叶斯、逻辑回归、随机森林）
   - ✅ BERT vs 传统模型性能对比

## 🚀 快速开始

### 前提条件

确保已训练好BERT模型：
```bash
python main.py
```

### 运行完整评估套件

```bash
bash run_complete_evaluation.sh
```

这将依次运行所有评估实验并生成完整报告。

## 📝 单独运行各个评估

### 1. 综合评估（推荐首先运行）

```bash
python comprehensive_evaluation.py --test-samples 1000 --output-dir evaluation_results
```

**参数说明：**
- `--test-samples`: 测试样本数量（可选，默认使用全部）
- `--output-dir`: 输出目录

**生成文件：**
- `confusion_matrix.png` - 混淆矩阵可视化
- `roc_curve.png` - ROC曲线
- `evaluation_report.txt` - 详细评估报告

**输出示例：**
```
准确率 (Accuracy):   0.8542
精确率 (Precision):  0.8621
召回率 (Recall):     0.8453
F1分数 (F1-score):   0.8536
AUC-ROC:             0.9234
```

### 2. K折交叉验证

```bash
python cross_validation.py --n-folds 5 --max-samples 5000 --output-dir evaluation_results
```

**参数说明：**
- `--n-folds`: K折交叉验证的折数（默认5）
- `--max-samples`: 最大样本数（减少训练时间）
- `--output-dir`: 输出目录

**生成文件：**
- `cross_validation_results.csv` - 各折详细结果
- `cross_validation_results.png` - 结果可视化图表
- `cross_validation_report.txt` - 交叉验证报告

**输出示例：**
```
平均指标 (± 标准差):
  准确率:  0.8512 ± 0.0123
  精确率:  0.8598 ± 0.0145
  召回率:  0.8431 ± 0.0167
  F1分数:  0.8513 ± 0.0134
```

### 3. 传统机器学习模型

```bash
python traditional_models.py --max-train-samples 10000 --save-models --output-dir evaluation_results
```

**参数说明：**
- `--max-features`: TF-IDF特征最大数量（默认5000）
- `--max-train-samples`: 训练样本数量限制
- `--save-models`: 保存训练好的模型
- `--output-dir`: 输出目录

**包含的模型：**
- 逻辑回归 (Logistic Regression)
- 支持向量机 (SVM - Linear Kernel)
- 朴素贝叶斯 (Naive Bayes)
- 随机森林 (Random Forest)

**生成文件：**
- `traditional_models_results.csv` - 结果数据
- `traditional_models_report.txt` - 详细报告
- `traditional_models/` - 保存的模型文件（如果使用 --save-models）

### 4. BERT vs 传统模型对比

```bash
python model_comparison.py --max-train-samples 10000 --output-dir evaluation_results
```

**参数说明：**
- `--max-train-samples`: 训练样本数量
- `--max-test-samples`: 测试样本数量（可选）
- `--output-dir`: 输出目录

**生成文件：**
- `model_comparison.png` - 性能指标对比图
- `time_comparison.png` - 时间效率对比图
- `model_comparison_report.txt` - 详细对比报告
- `model_comparison_results.csv` - 对比数据

## 📊 输出文件说明

所有评估结果默认保存在 `evaluation_results/` 目录下：

```
evaluation_results/
├── confusion_matrix.png              # 混淆矩阵
├── roc_curve.png                     # ROC曲线
├── evaluation_report.txt             # BERT综合评估报告
├── cross_validation_results.csv      # 交叉验证数据
├── cross_validation_results.png      # 交叉验证可视化
├── cross_validation_report.txt       # 交叉验证报告
├── traditional_models_results.csv    # 传统模型结果
├── traditional_models_report.txt     # 传统模型报告
├── model_comparison.png              # 模型对比图
├── time_comparison.png               # 时间对比图
├── model_comparison_report.txt       # 模型对比报告
├── model_comparison_results.csv      # 模型对比数据
└── traditional_models/               # 保存的传统模型
    ├── vectorizer.pkl
    ├── logistic_regression.pkl
    ├── svm_linear.pkl
    ├── naive_bayes.pkl
    └── random_forest.pkl
```

## 🎯 评估指标说明

### 准确率 (Accuracy)
正确预测的样本占总样本的比例。适用于类别平衡的数据集。

### 精确率 (Precision)
预测为正类中真正为正类的比例。衡量模型的"精准度"。

### 召回率 (Recall)
真正的正类中被正确预测的比例。衡量模型的"全面性"。

### F1分数 (F1-score)
精确率和召回率的调和平均数。综合考虑精确率和召回率的指标。

### AUC-ROC
ROC曲线下的面积，衡量模型区分正负类的能力。值越接近1表示性能越好。

### 混淆矩阵
- **真阴性 (TN)**: 正确预测为负类的数量
- **假阳性 (FP)**: 错误预测为正类的数量（第一类错误）
- **假阴性 (FN)**: 错误预测为负类的数量（第二类错误）
- **真阳性 (TP)**: 正确预测为正类的数量

## 💡 使用建议

### 1. 基础评估流程
```bash
# 步骤1：训练BERT模型
python main.py

# 步骤2：综合评估
python comprehensive_evaluation.py

# 步骤3：查看结果
cat evaluation_results/evaluation_report.txt
```

### 2. 鲁棒性测试
如果需要验证模型的稳定性和泛化能力：
```bash
python cross_validation.py --n-folds 5
```

### 3. 模型对比
如果需要与传统方法对比BERT的优势：
```bash
# 先训练传统模型
python traditional_models.py --save-models

# 再运行对比实验
python model_comparison.py
```

### 4. 完整评估
如果需要生成完整的评估报告：
```bash
bash run_complete_evaluation.sh
```

## ⚙️ 性能优化建议

### 减少评估时间
```bash
# 限制测试样本数量
python comprehensive_evaluation.py --test-samples 1000

# 限制交叉验证样本数量
python cross_validation.py --max-samples 3000

# 限制传统模型训练样本
python traditional_models.py --max-train-samples 5000
```

### GPU加速
如果有GPU，BERT评估会自动使用GPU加速。传统机器学习模型主要使用CPU。

## 📈 预期结果

### BERT模型典型性能
- 准确率: 85-90%
- F1分数: 84-89%
- AUC-ROC: 90-95%

### 传统模型典型性能
- SVM: F1 ~75-82%
- 逻辑回归: F1 ~73-80%
- 朴素贝叶斯: F1 ~70-78%
- 随机森林: F1 ~72-79%

### 性能差距
BERT相比最佳传统模型通常有5-10%的F1分数提升。

## 🔧 故障排除

### 1. 模型文件未找到
```
错误: 模型文件未找到: best_epoch_model.pth
解决: 先运行 python main.py 训练模型
```

### 2. 内存不足
```
解决: 减少样本数量
python comprehensive_evaluation.py --test-samples 500
```

### 3. 训练时间过长
```
解决: 使用较小的样本集进行交叉验证
python cross_validation.py --max-samples 2000
```

## 📚 参考资料

- [sklearn评估指标文档](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [BERT论文](https://arxiv.org/abs/1810.04805)
- [ROC-AUC解释](https://en.wikipedia.org/wiki/Receiver_operating_characteristic)

## ✨ 总结

本评估套件提供了全面的模型评估工具，包括：
- ✅ 多维度性能指标
- ✅ 可视化分析
- ✅ 鲁棒性测试
- ✅ 模型对比实验

使用这些工具可以全面了解BERT模型在情感分类任务上的表现，并与传统方法进行公平对比。
