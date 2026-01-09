#!/bin/bash

# 安装脚本

echo "=========================================="
echo "TextCNN Sentiment Classification - Setup"
echo "=========================================="
echo ""

echo "Installing dependencies..."
pip install -q torch numpy scikit-learn matplotlib tqdm

if [ $? -eq 0 ]; then
    echo "✅ Dependencies installed successfully!"
else
    echo "❌ Failed to install dependencies"
    exit 1
fi

echo ""
echo "=========================================="
echo "Setup completed!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Run 'python train.py' to train the model"
echo "  2. Run 'python predict.py' for interactive prediction"
echo "  3. Or run './run.sh' for complete workflow"


