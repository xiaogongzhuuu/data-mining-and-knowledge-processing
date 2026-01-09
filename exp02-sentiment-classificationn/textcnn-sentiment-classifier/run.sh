#!/bin/bash

# TextCNN 情感分类 - 完整运行脚本

echo "=========================================="
echo "TextCNN Sentiment Classification"
echo "=========================================="
echo ""

# 1. 训练模型
echo "Step 1: Training model..."
echo "------------------------------------------"
python train.py

if [ $? -ne 0 ]; then
    echo "❌ Training failed!"
    exit 1
fi

echo ""
echo "✅ Training completed!"
echo ""

# 2. 运行交互式预测（可选）
echo "Step 2: Interactive prediction (optional)"
echo "------------------------------------------"
echo "Run 'python predict.py' to try interactive prediction."
echo ""

echo "=========================================="
echo "All done! Check outputs/ for results."
echo "=========================================="


