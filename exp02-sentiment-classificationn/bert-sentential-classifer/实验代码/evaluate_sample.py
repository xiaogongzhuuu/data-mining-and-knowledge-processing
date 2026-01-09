import torch
import random
import argparse
import pandas as pd
from transformers import BertTokenizer
from config import Config
from model import SentimentClassifier
from load_data import DataLoader as DataLoaderClass
import os
from tqdm import tqdm

def load_model(device):
    """
    加载训练好的模型和分词器
    """
    config = Config()
    tokenizer = BertTokenizer.from_pretrained(config.model_name)
    model = SentimentClassifier(config.model_name, config.num_classes)
    
    model_path = config.model_save_path
    if os.path.exists(model_path):
        print(f"Loading weights from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print(f"Warning: Model file {model_path} not found. Using random weights.")
    
    model.to(device)
    model.eval()
    return model, tokenizer, config

def predict_batch(texts, model, tokenizer, device, config):
    """
    批量预测
    """
    encoded = tokenizer.batch_encode_plus(
        texts,
        max_length=config.max_seq_length,
        add_special_tokens=True,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    
    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        _, predictions = torch.max(outputs, dim=1)
        
    return predictions.cpu().numpy()

def main():
    parser = argparse.ArgumentParser(description='Evaluate model on test set sample')
    parser.add_argument('--samples', type=int, default=100, help='Number of samples to evaluate')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. 加载数据
    config = Config()
    data_loader = DataLoaderClass(config)
    print("Loading test data...")
    
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    test_file_path = os.path.join(current_dir, "test.csv")
    
    try:
        test_texts, test_labels = data_loader.load_csv(test_file_path)
    except FileNotFoundError:
        print(f"Error: {test_file_path} not found.")
        return

    # 2. 随机采样
    total_samples = len(test_texts)
    sample_size = min(args.samples, total_samples)
    print(f"Sampling {sample_size} examples from {total_samples} total test examples.")
    
    indices = random.sample(range(total_samples), sample_size)
    sampled_texts = [test_texts[i] for i in indices]
    sampled_labels = [test_labels[i] for i in indices]
    
    # 3. 加载模型
    model, tokenizer, config = load_model(device)
    
    # 4. 预测
    batch_size = config.batch_size
    all_predictions = []
    
    print("Running inference...")
    for i in tqdm(range(0, sample_size, batch_size)):
        batch_texts = sampled_texts[i:i+batch_size]
        predictions = predict_batch(batch_texts, model, tokenizer, device, config)
        all_predictions.extend(predictions)
        
    # 5. 计算准确率
    correct = sum(1 for p, l in zip(all_predictions, sampled_labels) if p == l)
    accuracy = correct / sample_size
    
    print("\n" + "="*50)
    print(f"Evaluation Results (Sample Size: {sample_size})")
    print("="*50)
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Correct: {correct}/{sample_size}")
    
    # 6. 展示错误案例
    print("\nError Analysis (First 5 errors):")
    error_count = 0
    for text, true_label, pred_label in zip(sampled_texts, sampled_labels, all_predictions):
        if true_label != pred_label:
            error_count += 1
            print(f"\n[Error #{error_count}]")
            print(f"Text: {text[:100]}...")
            print(f"True: {'Positive' if true_label==1 else 'Negative'} | Pred: {'Positive' if pred_label==1 else 'Negative'}")
            if error_count >= 5:
                break

if __name__ == "__main__":
    main()
