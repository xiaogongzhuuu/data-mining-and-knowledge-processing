import torch
import random
import argparse
import os
from transformers import BertTokenizer
from config import Config
from model import SentimentClassifier
from load_data import DataLoader as DataLoaderClass
from tqdm import tqdm

def load_trained_model(device, config):
    """
    加载训练好的模型
    """
    model = SentimentClassifier(config.model_name, config.num_classes)
    model_path = config.model_save_path

    if os.path.exists(model_path):
        print(f"✓ Loading trained model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        raise FileNotFoundError(f"Trained model not found at {model_path}")

    model.to(device)
    model.eval()
    return model

def load_untrained_model(device, config):
    """
    创建未训练的模型（随机初始化的分类层）
    """
    print(f"✓ Creating untrained model (random classifier weights)")
    model = SentimentClassifier(config.model_name, config.num_classes)
    model.to(device)
    model.eval()
    return model

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

def evaluate_model(model, texts, labels, tokenizer, device, config, batch_size):
    """
    评估模型准确率
    """
    all_predictions = []
    sample_size = len(texts)

    for i in tqdm(range(0, sample_size, batch_size), desc="Evaluating"):
        batch_texts = texts[i:i+batch_size]
        predictions = predict_batch(batch_texts, model, tokenizer, device, config)
        all_predictions.extend(predictions)

    correct = sum(1 for p, l in zip(all_predictions, labels) if p == l)
    accuracy = correct / sample_size

    return accuracy, correct, sample_size, all_predictions

def main():
    parser = argparse.ArgumentParser(description='Compare trained vs untrained model accuracy')
    parser.add_argument('--samples', type=int, default=1000,
                        help='Number of test samples to evaluate (default: 1000)')
    parser.add_argument('--full-test', action='store_true',
                        help='Evaluate on full test set (ignore --samples)')
    args = parser.parse_args()

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # 加载配置
    config = Config()

    # 加载tokenizer
    tokenizer = BertTokenizer.from_pretrained(config.model_name)

    # 加载测试数据
    print("Loading test data...")
    data_loader = DataLoaderClass(config)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    test_file_path = os.path.join(current_dir, "test.csv")

    try:
        test_texts, test_labels = data_loader.load_csv(test_file_path)
    except FileNotFoundError:
        print(f"Error: {test_file_path} not found.")
        return

    # 采样测试数据
    total_samples = len(test_texts)
    if args.full_test:
        sample_size = total_samples
        sampled_texts = test_texts
        sampled_labels = test_labels
        print(f"Evaluating on FULL test set: {sample_size} samples\n")
    else:
        sample_size = min(args.samples, total_samples)
        print(f"Sampling {sample_size} examples from {total_samples} total test examples\n")
        indices = random.sample(range(total_samples), sample_size)
        sampled_texts = [test_texts[i] for i in indices]
        sampled_labels = [test_labels[i] for i in indices]

    # ==================== 评估未训练模型 ====================
    print("=" * 60)
    print("1️⃣  EVALUATING UNTRAINED MODEL (Random Weights)")
    print("=" * 60)

    untrained_model = load_untrained_model(device, config)
    untrained_acc, untrained_correct, _, untrained_preds = evaluate_model(
        untrained_model, sampled_texts, sampled_labels, tokenizer, device, config, config.batch_size
    )

    print(f"\n📊 Untrained Model Results:")
    print(f"   Accuracy: {untrained_acc:.2%}")
    print(f"   Correct: {untrained_correct}/{sample_size}")

    # 清理内存
    del untrained_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ==================== 评估训练后模型 ====================
    print("\n" + "=" * 60)
    print("2️⃣  EVALUATING TRAINED MODEL")
    print("=" * 60)

    trained_model = load_trained_model(device, config)
    trained_acc, trained_correct, _, trained_preds = evaluate_model(
        trained_model, sampled_texts, sampled_labels, tokenizer, device, config, config.batch_size
    )

    print(f"\n📊 Trained Model Results:")
    print(f"   Accuracy: {trained_acc:.2%}")
    print(f"   Correct: {trained_correct}/{sample_size}")

    # ==================== 对比结果 ====================
    print("\n" + "=" * 60)
    print("📈 COMPARISON RESULTS")
    print("=" * 60)
    print(f"Test Samples:         {sample_size}")
    print(f"Untrained Accuracy:   {untrained_acc:.2%}  ({untrained_correct}/{sample_size})")
    print(f"Trained Accuracy:     {trained_acc:.2%}  ({trained_correct}/{sample_size})")
    print(f"Improvement:          {(trained_acc - untrained_acc):.2%}")
    print(f"Relative Improvement: {((trained_acc - untrained_acc) / untrained_acc * 100):.1f}%")
    print("=" * 60)

    # ==================== 错误案例分析 ====================
    print("\n📝 Error Analysis - Trained Model (First 5 errors):")
    error_count = 0
    for text, true_label, pred_label in zip(sampled_texts, sampled_labels, trained_preds):
        if true_label != pred_label:
            error_count += 1
            print(f"\n[Error #{error_count}]")
            print(f"Text: {text[:150]}...")
            print(f"True: {'Positive (1)' if true_label==1 else 'Negative (0)'} | "
                  f"Pred: {'Positive (1)' if pred_label==1 else 'Negative (0)'}")
            if error_count >= 5:
                break

    if error_count == 0:
        print("No errors found! Perfect accuracy on this sample.")

if __name__ == "__main__":
    main()
