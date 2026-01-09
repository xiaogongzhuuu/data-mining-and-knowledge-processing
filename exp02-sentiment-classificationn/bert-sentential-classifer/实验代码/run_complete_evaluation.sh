#!/bin/bash
# BERT情感分类实验 - 完整评估套件统一运行脚本

echo "========================================"
echo "BERT情感分类 - 完整评估套件"
echo "========================================"
echo ""

# 颜色定义
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 检查是否有训练好的模型
if [ ! -f "best_epoch_model.pth" ]; then
    echo -e "${YELLOW}警告: 未找到训练好的BERT模型 best_epoch_model.pth${NC}"
    echo "请先运行 python main.py 训练模型"
    exit 1
fi

echo -e "${GREEN}找到训练好的BERT模型${NC}"
echo ""

# 创建输出目录
OUTPUT_DIR="evaluation_results"
mkdir -p $OUTPUT_DIR
echo "输出目录: $OUTPUT_DIR"
echo ""

# ============================================
# 1. 综合评估（Accuracy, Precision, Recall, F1, AUC-ROC, 混淆矩阵）
# ============================================
echo -e "${BLUE}========================================"
echo "1. BERT模型综合评估"
echo "   (Accuracy, Precision, Recall, F1, AUC-ROC, 混淆矩阵)"
echo -e "========================================${NC}"
python comprehensive_evaluation.py --output-dir $OUTPUT_DIR
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ 综合评估完成${NC}"
else
    echo -e "${RED}✗ 综合评估失败${NC}"
fi
echo ""

# ============================================
# 2. K折交叉验证（鲁棒性测试）
# ============================================
echo -e "${BLUE}========================================"
echo "2. K折交叉验证（鲁棒性测试）"
echo -e "========================================${NC}"
echo -e "${YELLOW}注意: 此步骤需要训练多个模型，可能需要较长时间${NC}"
read -p "是否继续? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    python cross_validation.py --n-folds 5 --max-samples 5000 --output-dir $OUTPUT_DIR
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ 交叉验证完成${NC}"
    else
        echo -e "${RED}✗ 交叉验证失败${NC}"
    fi
else
    echo "跳过交叉验证"
fi
echo ""

# ============================================
# 3. 传统机器学习模型评估
# ============================================
echo -e "${BLUE}========================================"
echo "3. 传统机器学习模型评估"
echo "   (SVM, 朴素贝叶斯, 逻辑回归, 随机森林)"
echo -e "========================================${NC}"
python traditional_models.py --max-train-samples 10000 --save-models --output-dir $OUTPUT_DIR
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ 传统模型评估完成${NC}"
else
    echo -e "${RED}✗ 传统模型评估失败${NC}"
fi
echo ""

# ============================================
# 4. BERT vs 传统模型对比
# ============================================
echo -e "${BLUE}========================================"
echo "4. BERT vs 传统模型对比实验"
echo -e "========================================${NC}"
python model_comparison.py --max-train-samples 10000 --output-dir $OUTPUT_DIR
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ 模型对比完成${NC}"
else
    echo -e "${RED}✗ 模型对比失败${NC}"
fi
echo ""

# ============================================
# 总结
# ============================================
echo -e "${GREEN}========================================"
echo "评估套件运行完成！"
echo -e "========================================${NC}"
echo ""
echo "生成的文件 (在 $OUTPUT_DIR 目录下):"
echo ""
echo "【综合评估】"
echo "  • confusion_matrix.png           - 混淆矩阵"
echo "  • roc_curve.png                  - ROC曲线"
echo "  • evaluation_report.txt          - 详细评估报告"
echo ""
echo "【交叉验证】（如果运行）"
echo "  • cross_validation_results.csv   - 各折结果"
echo "  • cross_validation_results.png   - 结果可视化"
echo "  • cross_validation_report.txt    - 交叉验证报告"
echo ""
echo "【传统模型】"
echo "  • traditional_models_results.csv - 传统模型结果"
echo "  • traditional_models_report.txt  - 传统模型报告"
echo "  • traditional_models/            - 保存的模型文件"
echo ""
echo "【模型对比】"
echo "  • model_comparison.png           - 模型性能对比图"
echo "  • time_comparison.png            - 时间效率对比图"
echo "  • model_comparison_report.txt    - 对比分析报告"
echo "  • model_comparison_results.csv   - 对比结果数据"
echo ""
echo -e "${GREEN}所有评估结果已保存！${NC}"
