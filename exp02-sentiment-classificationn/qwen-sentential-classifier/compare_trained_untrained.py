"""
训练前后模型对比 - Qwen情感分类器版本
对比未训练模型和已训练模型的准确率差异
"""
import torch
import random
import argparse
import os
from transformers import AutoTokenizer
from config import Config
from model import SentimentClassifier
from load_data import DataLoader as DataLoaderClass
from tqdm import tqdm

def load_trained_model(device, config):
    """加载训练好的模型"""
    model = SentimentClassifier(config.model_name, config.num_classes, freeze_base=True)
    model_path = config.model_save_path

    if os.path.exists(model_path):
        print(f"✓ 加载训练好的模型: {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        raise FileNotFoundError(f"未找到训练好的模型: {model_path}")

    model.to(device)
    model.eval()
    return model

def load_untrained_model(device, config):
    """创建未训练的模型（随机初始化的分类层）"""
    print(f"✓ 创建未训练模型（随机分类器权重）")
    model = SentimentClassifier(config.model_name, config.num_classes, freeze_base=True)
    model.to(device)
    model.eval()
    return model

def predict_batch(texts, model, tokenizer, device, config):
    """批量预测"""
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
    """评估模型准确率"""
    all_predictions = []
    sample_size = len(texts)

    for i in tqdm(range(0, sample_size, batch_size), desc="评估中"):
        batch_texts = texts[i:i+batch_size]
        predictions = predict_batch(batch_texts, model, tokenizer, device, config)
        all_predictions.extend(predictions)

    correct = sum(1 for p, l in zip(all_predictions, labels) if p == l)
    accuracy = correct / sample_size

    return accuracy, correct, sample_size, all_predictions

def main():
    parser = argparse.ArgumentParser(description='对比训练前后模型准确率')
    parser.add_argument('--samples', type=int, default=1000,
                        help='测试样本数量 (默认: 1000)')
    parser.add_argument('--full-test', action='store_true',
                        help='评估完整测试集（忽略--samples）')
    args = parser.parse_args()

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}\n")

    # 加载配置
    config = Config()

    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name,
        trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载测试数据
    print("加载测试数据...")
    data_loader = DataLoaderClass(config)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    test_file_path = os.path.join(current_dir, "test.csv")

    try:
        test_texts, test_labels = data_loader.load_csv(test_file_path)
    except FileNotFoundError:
        print(f"错误: 未找到 {test_file_path}")
        return

    # 采样测试数据
    total_samples = len(test_texts)
    if args.full_test:
        sample_size = total_samples
        sampled_texts = test_texts
        sampled_labels = test_labels
        print(f"评估完整测试集: {sample_size} 样本\n")
    else:
        sample_size = min(args.samples, total_samples)
        print(f"从 {total_samples} 个样本中随机抽取 {sample_size} 个\n")
        indices = random.sample(range(total_samples), sample_size)
        sampled_texts = [test_texts[i] for i in indices]
        sampled_labels = [test_labels[i] for i in indices]

    # ==================== 评估未训练模型 ====================
    print("=" * 60)
    print("1️⃣  评估未训练模型（随机权重）")
    print("=" * 60)

    untrained_model = load_untrained_model(device, config)
    untrained_acc, untrained_correct, _, untrained_preds = evaluate_model(
        untrained_model, sampled_texts, sampled_labels, tokenizer, device, config, config.batch_size
    )

    print(f"\n📊 未训练模型结果:")
    print(f"   准确率: {untrained_acc:.2%}")
    print(f"   正确数: {untrained_correct}/{sample_size}")

    # 清理内存
    del untrained_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ==================== 评估训练后模型 ====================
    print("\n" + "=" * 60)
    print("2️⃣  评估训练后模型")
    print("=" * 60)

    trained_model = load_trained_model(device, config)
    trained_acc, trained_correct, _, trained_preds = evaluate_model(
        trained_model, sampled_texts, sampled_labels, tokenizer, device, config, config.batch_size
    )

    print(f"\n📊 训练后模型结果:")
    print(f"   准确率: {trained_acc:.2%}")
    print(f"   正确数: {trained_correct}/{sample_size}")

    # ==================== 对比结果 ====================
    print("\n" + "=" * 60)
    print("📈 对比结果")
    print("=" * 60)
    print(f"测试样本数:       {sample_size}")
    print(f"未训练准确率:     {untrained_acc:.2%}  ({untrained_correct}/{sample_size})")
    print(f"训练后准确率:     {trained_acc:.2%}  ({trained_correct}/{sample_size})")
    print(f"准确率提升:       {(trained_acc - untrained_acc):.2%}")
    if untrained_acc > 0:
        print(f"相对提升:         {((trained_acc - untrained_acc) / untrained_acc * 100):.1f}%")
    print("=" * 60)

    # ==================== 错误案例分析 ====================
    print("\n📝 错误案例分析 - 训练后模型（前5个错误）:")
    error_count = 0
    for text, true_label, pred_label in zip(sampled_texts, sampled_labels, trained_preds):
        if true_label != pred_label:
            error_count += 1
            print(f"\n[错误 #{error_count}]")
            print(f"文本: {text[:150]}...")
            print(f"真实: {'正面 (1)' if true_label==1 else '负面 (0)'} | "
                  f"预测: {'正面 (1)' if pred_label==1 else '负面 (0)'}")
            if error_count >= 5:
                break

    if error_count == 0:
        print("未发现错误！准确率100%")

if __name__ == "__main__":
    main()
