"""
训练脚本：对 Qwen2.5-0.5B 做情感分类微调（适配 8GB GPU）
- 自动读取当前目录下的 train.csv, dev.csv, test.csv
- 使用 Hugging Face tokenizer + AutoModel (trust_remote_code=True)
- 自定义分类头 + 训练循环
"""

import os
import csv
import math
import random
from typing import List, Tuple

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
import json

# ----------------- 配置项 -----------------
MODEL_NAME = "Qwen/Qwen2.5-0.5B"
CACHE_DIR = "../hf_cache"  # 使用本地缓存目录
OUTPUT_DIR = "./qwen_sentiment_out"   # 输出目录
LOG_DIR = "./logs"                     # 日志目录

MAX_LENGTH = 128                      # 文本截断长度（根据GPU显存调整：48/128/256）
MAX_TRAIN_SAMPLES = 2000              # 训练集最大样本数（快速实验可降到 500-1000）

per_device_batch_size = 4             # 每卡 batch（显存不足降到 1-2）
gradient_accumulation_steps = 4        # 梯度累积步数
num_epochs = 3                        # 训练轮数
learning_rate = 3e-5                  # 学习率
weight_decay = 0.01                   # 权重衰减
seed = 42                             # 随机种子
# ------------------------------------------

# 创建必要的目录
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️  Using device: {device}")


# 固定随机种子
def set_seed(s=seed):
    random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


set_seed()


# --------- 数据加载 ---------
def load_csv_data(filepath: str) -> Tuple[List[str], List[int]]:
    """
    加载 CSV 格式的数据文件
    格式：label,text
    返回：texts, labels
    """
    texts, labels = [], []
    
    if not os.path.exists(filepath):
        print(f"⚠️  Warning: {filepath} not found")
        return texts, labels
    
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        reader = csv.reader(f)
        next(reader, None)  # 跳过表头（如果有）
        
        for row in reader:
            if len(row) < 2:
                continue
            
            try:
                label = int(row[0].strip())
                text = row[1].strip() if len(row) > 1 else ""
                
                # 合并多列文本（处理文本中包含逗号的情况）
                if len(row) > 2:
                    text = " ".join(r.strip() for r in row[1:] if r.strip())
                
                text = text.replace("\n", " ").replace("\r", " ").strip()
                
                if not text:
                    continue
                
                # 统一 label 到 0/1（假设 1=负面, 2=正面）
                mapped_label = 0 if label == 1 else 1
                
                texts.append(text)
                labels.append(mapped_label)
                
            except (ValueError, IndexError):
                continue
    
    return texts, labels


def load_dataset():
    """加载训练、验证和测试数据"""
    print("📂 Loading datasets...")
    
    train_texts, train_labels = load_csv_data("../train.csv")
    dev_texts, dev_labels = load_csv_data("../dev.csv")
    test_texts, test_labels = load_csv_data("../test.csv")
    
    if len(train_texts) == 0:
        raise RuntimeError("❌ No training data found. Check train.csv")
    
    # 如果没有验证集，从训练集切分
    if len(dev_texts) == 0 and len(train_texts) >= 20:
        n_dev = max(int(0.1 * len(train_texts)), 20)
        dev_texts = train_texts[:n_dev]
        dev_labels = train_labels[:n_dev]
        train_texts = train_texts[n_dev:]
        train_labels = train_labels[n_dev:]
        print(f"📊 Split {n_dev} samples from train as dev set")
    
    # 随机下采样训练集（核心加速点）
    if len(train_texts) > MAX_TRAIN_SAMPLES:
        rng = random.Random(42)
        indices = list(range(len(train_texts)))
        rng.shuffle(indices)
        indices = indices[:MAX_TRAIN_SAMPLES]
        
        train_texts = [train_texts[i] for i in indices]
        train_labels = [train_labels[i] for i in indices]
        
        print(f"🚀 Randomly sampled train set to {MAX_TRAIN_SAMPLES}")
    
    # 打印统计信息
    def label_stats(labels):
        if not labels:
            return {"total": 0, "pos": 0, "neg": 0}
        total = len(labels)
        pos = sum(labels)
        return {"total": total, "pos": pos, "neg": total - pos}
    
    print("\n📊 Dataset Statistics:")
    print(f"  Train: {label_stats(train_labels)}")
    print(f"  Dev  : {label_stats(dev_labels)}")
    print(f"  Test : {label_stats(test_labels)}")
    print()
    
    return (
        train_texts, train_labels,
        dev_texts, dev_labels,
        test_texts, test_labels
    )


# --------- Dataset & DataLoader ---------
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        item = {k: v.squeeze(0) for k, v in encoding.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def make_dataloader(texts, labels, tokenizer, batch_size, shuffle=True):
    if not texts:
        return None
    dataset = SentimentDataset(texts, labels, tokenizer, max_length=MAX_LENGTH)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# --------- Model: Qwen + Classification Head ---------
class QwenForSequenceClassification(nn.Module):
    def __init__(self, model_name, num_labels=2, cache_dir=None):
        super().__init__()
        
        print(f"🔄 Loading {model_name}...")
        self.base = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            cache_dir=cache_dir,
            low_cpu_mem_usage=True
        )
        
        # 获取隐藏层大小
        hidden_size = None
        cfg = getattr(self.base, "config", None)
        if cfg:
            hidden_size = getattr(cfg, "hidden_size", None) or getattr(cfg, "d_model", None)
        
        self.num_labels = num_labels
        self.classifier = None
        
        if hidden_size is not None:
            self.classifier = nn.Linear(hidden_size, num_labels)
            print(f"✅ Model loaded (hidden_size={hidden_size})")
    
    def forward(self, input_ids, attention_mask, labels=None):
        # 获取 base model 输出
        outputs = self.base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            return_dict=True
        )
        
        # 提取最后一层隐藏状态
        if hasattr(outputs, "last_hidden_state"):
            last_hidden = outputs.last_hidden_state
        elif isinstance(outputs, (tuple, list)) and len(outputs) > 0:
            last_hidden = outputs[0]
        else:
            raise RuntimeError("Cannot extract last_hidden_state from model output")
        
        # Mean pooling（平均池化，忽略 padding）
        mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
        summed = (last_hidden * mask).sum(1)
        counts = mask.sum(1).clamp(min=1e-9)
        pooled = summed / counts
        
        # Lazy initialization of classifier
        if self.classifier is None:
            hidden_size = pooled.size(-1)
            self.classifier = nn.Linear(hidden_size, self.num_labels).to(pooled.device)
            print(f"✅ Classifier initialized (hidden_size={hidden_size})")
        
        logits = self.classifier(pooled)
        
        # 计算损失
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return {"loss": loss, "logits": logits}
        
        return {"logits": logits}


# --------- Evaluation ---------
def evaluate_model(model, dataloader, device):
    """评估模型性能"""
    if dataloader is None:
        return {}
    
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs["logits"]
            
            probs = torch.softmax(logits, dim=-1)[:, 1]
            preds = logits.argmax(dim=-1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0
    cm = confusion_matrix(all_labels, all_preds).tolist()
    
    return {
        "acc": float(acc),
        "f1": float(f1),
        "auc": float(auc),
        "confusion_matrix": cm
    }


# --------- Main Training Loop ---------
def main():
    print("="*60)
    print("🚀 Qwen2.5 Sentiment Classification Training")
    print("="*60)
    
    # 1. 加载数据
    train_texts, train_labels, dev_texts, dev_labels, test_texts, test_labels = load_dataset()
    
    # 2. 加载 tokenizer
    print(f"🔄 Loading tokenizer from {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        cache_dir=CACHE_DIR
    )
    
    # 确保有 pad_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("⚠️  Set pad_token = eos_token")
    
    # 3. 创建 DataLoader
    train_loader = make_dataloader(train_texts, train_labels, tokenizer, per_device_batch_size, shuffle=True)
    dev_loader = make_dataloader(dev_texts, dev_labels, tokenizer, per_device_batch_size, shuffle=False)
    test_loader = make_dataloader(test_texts, test_labels, tokenizer, per_device_batch_size, shuffle=False)
    
    # 4. 创建模型
    model = QwenForSequenceClassification(MODEL_NAME, num_labels=2, cache_dir=CACHE_DIR)
    model.to(device)
    
    # 调整 token embeddings（如果添加了新 token）
    if hasattr(model.base, "resize_token_embeddings"):
        model.base.resize_token_embeddings(len(tokenizer))
    
    # 5. 优化器和调度器
    no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0
        },
    ]
    
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=learning_rate)
    
    total_steps = (len(train_loader) // gradient_accumulation_steps) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, total_steps // 10),
        num_training_steps=max(1, total_steps)
    )
    
    # 6. 日志文件
    log_file = os.path.join(LOG_DIR, f"qwen25_n{len(train_texts)}_seed{seed}.json")
    history = []
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            history = json.load(f)
    
    # 7. 训练循环
    print("\n" + "="*60)
    print("🏋️  Starting Training...")
    print("="*60)
    
    best_dev_f1 = 0.0
    global_step = 0
    
    for epoch in range(num_epochs):
        print(f"\n📅 Epoch {epoch+1}/{num_epochs}")
        model.train()
        
        optimizer.zero_grad()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for step, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs["loss"] / gradient_accumulation_steps
            loss.backward()
            running_loss += loss.item()
            
            if (step + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                
                pbar.set_postfix({"loss": f"{running_loss:.4f}"})
                running_loss = 0.0
        
        # 8. 验证和测试
        print("\n📊 Evaluating...")
        dev_metrics = evaluate_model(model, dev_loader, device)
        test_metrics = evaluate_model(model, test_loader, device)
        
        if dev_metrics:
            print(f"  Dev  | Acc: {dev_metrics['acc']:.4f} | F1: {dev_metrics['f1']:.4f} | AUC: {dev_metrics['auc']:.4f}")
        if test_metrics:
            print(f"  Test | Acc: {test_metrics['acc']:.4f} | F1: {test_metrics['f1']:.4f} | AUC: {test_metrics['auc']:.4f}")
        
        # 9. 保存日志
        log_entry = {
            "epoch": epoch + 1,
            "sample_size": len(train_texts),
            "seed": seed,
            "dev_acc": dev_metrics.get("acc"),
            "dev_f1": dev_metrics.get("f1"),
            "dev_auc": dev_metrics.get("auc"),
            "test_acc": test_metrics.get("acc"),
            "test_f1": test_metrics.get("f1"),
            "test_auc": test_metrics.get("auc"),
            "confusion_matrix": dev_metrics.get("confusion_matrix")
        }
        
        history.append(log_entry)
        with open(log_file, 'w') as f:
            json.dump(history, f, indent=2)
        
        # 10. 保存检查点
        ckpt_dir = os.path.join(OUTPUT_DIR, f"checkpoint-epoch{epoch+1}")
        os.makedirs(ckpt_dir, exist_ok=True)
        
        try:
            model.base.save_pretrained(ckpt_dir)
        except Exception as e:
            print(f"⚠️  Cannot save base model with save_pretrained: {e}")
            torch.save(model.base.state_dict(), os.path.join(ckpt_dir, "base_state_dict.pt"))
        
        torch.save(model.classifier.state_dict(), os.path.join(ckpt_dir, "classifier.pt"))
        tokenizer.save_pretrained(ckpt_dir)
        
        # 保存最佳模型
        if dev_metrics and dev_metrics['f1'] > best_dev_f1:
            best_dev_f1 = dev_metrics['f1']
            best_ckpt_dir = os.path.join(OUTPUT_DIR, "best_model")
            os.makedirs(best_ckpt_dir, exist_ok=True)
            
            try:
                model.base.save_pretrained(best_ckpt_dir)
            except:
                torch.save(model.base.state_dict(), os.path.join(best_ckpt_dir, "base_state_dict.pt"))
            
            torch.save(model.classifier.state_dict(), os.path.join(best_ckpt_dir, "classifier.pt"))
            tokenizer.save_pretrained(best_ckpt_dir)
            print(f"💾 Best model saved (F1: {best_dev_f1:.4f})")
    
    print("\n" + "="*60)
    print("✅ Training completed!")
    print(f"📊 Best Dev F1: {best_dev_f1:.4f}")
    print(f"💾 Models saved to: {OUTPUT_DIR}")
    print(f"📝 Training log: {log_file}")
    print("="*60)


if __name__ == "__main__":
    main()

