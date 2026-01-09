"""
预测脚本：使用训练好的 TextCNN 模型进行预测
"""

import torch
import numpy as np

import config
from data_loader import Vocabulary, simple_tokenize
from model import create_model


class SentimentPredictor:
    """情感预测器"""
    
    def __init__(self, model_path: str, vocab_path: str, device: str = "cpu"):
        """
        初始化预测器
        
        Args:
            model_path: 模型权重路径
            vocab_path: 词表路径
            device: 设备
        """
        self.device = torch.device(device)
        
        # 加载词表
        print(f"Loading vocabulary from {vocab_path}...")
        self.vocab = Vocabulary.load(vocab_path)
        
        # 加载模型
        print(f"Loading model from {model_path}...")
        self.model = create_model(vocab_size=len(self.vocab))
        
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✓ Model loaded (Dev F1: {checkpoint.get('dev_f1', 'N/A')})")
        
        self.label_names = ["负面 (Negative)", "正面 (Positive)"]
    
    def predict(self, text: str, return_prob: bool = False):
        """
        预测单个文本的情感
        
        Args:
            text: 输入文本
            return_prob: 是否返回概率
        
        Returns:
            prediction: 预测的类别 (0: 负面, 1: 正面)
            prob: 预测概率（如果 return_prob=True）
        """
        # 文本预处理和编码
        indices = self.vocab.encode(text)
        
        # 截断或填充
        if len(indices) > config.MAX_SEQ_LENGTH:
            indices = indices[:config.MAX_SEQ_LENGTH]
        else:
            indices = indices + [self.vocab.word2idx[self.vocab.pad_token]] * (config.MAX_SEQ_LENGTH - len(indices))
        
        # 转换为tensor
        input_ids = torch.tensor([indices], dtype=torch.long).to(self.device)
        
        # 预测
        with torch.no_grad():
            logits = self.model(input_ids)
            probs = torch.softmax(logits, dim=1)
            pred = torch.argmax(logits, dim=1).item()
            confidence = probs[0][pred].item()
        
        if return_prob:
            return pred, confidence, probs[0].cpu().numpy()
        else:
            return pred, confidence
    
    def predict_batch(self, texts: list):
        """
        批量预测
        
        Args:
            texts: 文本列表
        
        Returns:
            predictions: 预测列表
            confidences: 置信度列表
        """
        predictions = []
        confidences = []
        
        for text in texts:
            pred, conf = self.predict(text)
            predictions.append(pred)
            confidences.append(conf)
        
        return predictions, confidences


def interactive_demo():
    """交互式演示"""
    print("\n" + "="*80)
    print("TextCNN Sentiment Analysis - Interactive Demo")
    print("="*80)
    
    # 初始化预测器
    predictor = SentimentPredictor(
        model_path=config.MODEL_SAVE_PATH,
        vocab_path=config.VOCAB_SAVE_PATH,
        device=config.DEVICE if torch.cuda.is_available() else "cpu"
    )
    
    print("\n💡 Enter a review text to analyze its sentiment.")
    print("   Type 'quit' or 'exit' to stop.\n")
    
    # 示例文本
    examples = [
        "This product is amazing! I love it so much. Highly recommended!",
        "Terrible quality. Broke after 2 days. Don't waste your money.",
        "It's okay, nothing special. Average product.",
        "Best purchase ever! Exceeded all my expectations.",
        "Disappointed. Not as described. Would not buy again."
    ]
    
    print("="*80)
    print("📝 Example Predictions:")
    print("="*80)
    
    for i, text in enumerate(examples, 1):
        pred, confidence, probs = predictor.predict(text, return_prob=True)
        
        print(f"\n{i}. Text: \"{text}\"")
        print(f"   Prediction: {predictor.label_names[pred]}")
        print(f"   Confidence: {confidence:.4f}")
        print(f"   Probabilities: [Neg: {probs[0]:.4f}, Pos: {probs[1]:.4f}]")
    
    print("\n" + "="*80)
    print("🎮 Interactive Mode")
    print("="*80 + "\n")
    
    while True:
        try:
            text = input("Enter review text: ").strip()
            
            if text.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            if not text:
                continue
            
            pred, confidence, probs = predictor.predict(text, return_prob=True)
            
            print(f"\n  📊 Prediction: {predictor.label_names[pred]}")
            print(f"  📈 Confidence: {confidence:.4f}")
            print(f"  📉 Probabilities: [Neg: {probs[0]:.4f}, Pos: {probs[1]:.4f}]\n")
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")


if __name__ == "__main__":
    interactive_demo()

