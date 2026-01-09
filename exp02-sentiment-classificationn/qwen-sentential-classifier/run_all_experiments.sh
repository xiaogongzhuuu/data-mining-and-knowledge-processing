#!/bin/bash
# Qwen情感分类实验分析套件 - 统一运行脚本
# 配置：Qwen2-0.5B, batch_size=16, lr=1e-5, epochs=3, max_seq_length=256

echo "========================================"
echo "Qwen情感分类实验分析套件"
echo "模型: Qwen2-0.5B (冻结基础模型)"
echo "========================================"
echo ""

# 颜色定义
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 检查是否有训练好的模型
if [ ! -d "saved_models" ] || [ ! -f "saved_models/sentiment_model.pth" ]; then
    echo -e "${YELLOW}警告: 未找到训练好的模型 saved_models/sentiment_model.pth${NC}"
    echo "请先运行 python main.py 训练模型"
    exit 1
fi

echo -e "${GREEN}找到训练好的模型${NC}"
echo ""

# 1. 训练前后对比
echo -e "${BLUE}========================================"
echo "1. 训练前后模型对比"
echo -e "========================================${NC}"
python compare_trained_untrained.py --samples 500
echo ""

# 2. 测试集抽样稳定性分析
echo -e "${BLUE}========================================"
echo "2. 测试集抽样稳定性分析"
echo -e "========================================${NC}"
python sample_stability_analysis.py --sample-sizes 50 100 200 500 --trials 5
echo ""

# 3. 训练集大小影响分析
echo -e "${BLUE}========================================"
echo "3. 训练集大小影响分析"
echo -e "========================================${NC}"
echo -e "${YELLOW}注意: 此步骤需要重新训练多个模型，可能需要较长时间${NC}"
read -p "是否继续? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    python train_size_analysis.py --train-sizes 100 500 1000 2000 --epochs 3 --val-size 500
else
    echo "跳过训练集大小分析"
fi
echo ""

# 4. 训练轮数影响分析
echo -e "${BLUE}========================================"
echo "4. 训练轮数影响分析"
echo -e "========================================${NC}"
echo -e "${YELLOW}注意: 此步骤需要训练多个epochs，可能需要较长时间${NC}"
read -p "是否继续? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    python epoch_analysis.py --max-epochs 5 --train-size 2000 --val-size 500 --test-size 500
else
    echo "跳过训练轮数分析"
fi
echo ""

echo -e "${GREEN}========================================"
echo "实验分析完成！"
echo -e "========================================${NC}"
echo "生成的文件:"
echo "  - compare_trained_untrained.py 的输出"
echo "  - sample_stability_results.csv"
echo "  - sample_stability_plot.png"
echo "  - train_size_results.csv"
echo "  - train_size_analysis.png"
echo "  - epoch_analysis_results.csv"
echo "  - epoch_analysis_learning_curves.png"
echo "  - epoch_analysis_overfitting.png"
echo ""
