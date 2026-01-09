"""
Qwen模型兼容性检查脚本
在修改config.py之前运行此脚本，确保Qwen模型可以正常加载
"""
import torch
from transformers import AutoTokenizer, AutoModel
import warnings
warnings.filterwarnings('ignore')

def test_qwen_model(model_name):
    """测试Qwen模型是否可以正常加载和使用"""

    print(f"\n{'='*60}")
    print(f"测试模型: {model_name}")
    print(f"{'='*60}\n")

    try:
        # 1. 测试Tokenizer加载
        print("1️⃣  测试Tokenizer加载...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True  # Qwen需要这个参数
        )
        print(f"   ✅ Tokenizer加载成功")
        print(f"   - Vocab size: {tokenizer.vocab_size}")
        print(f"   - Has pad_token: {tokenizer.pad_token is not None}")

        # 如果没有pad_token，设置为eos_token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            print(f"   - 设置 pad_token = eos_token")

        # 2. 测试Model加载
        print("\n2️⃣  测试Model加载...")
        model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True  # Qwen需要这个参数
        )
        print(f"   ✅ Model加载成功")
        print(f"   - Hidden size: {model.config.hidden_size}")
        print(f"   - Num layers: {model.config.num_hidden_layers}")

        # 3. 测试是否有pooler_output
        print("\n3️⃣  测试模型输出格式...")
        test_text = "这是一个测试文本"
        inputs = tokenizer(test_text, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)

        has_pooler = hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None
        print(f"   - Has pooler_output: {has_pooler}")
        print(f"   - last_hidden_state shape: {outputs.last_hidden_state.shape}")

        if not has_pooler:
            print(f"   ℹ️  Qwen没有pooler_output，将使用 last_hidden_state[:, 0, :]")
            print(f"   ✅ 代码已经处理了这种情况（model.py第52-53行）")

        # 4. 测试分类器兼容性
        print("\n4️⃣  测试分类器兼容性...")
        num_classes = 2
        classifier = torch.nn.Linear(model.config.hidden_size, num_classes)

        if has_pooler:
            pooled = outputs.pooler_output
        else:
            pooled = outputs.last_hidden_state[:, 0, :]

        logits = classifier(pooled)
        print(f"   ✅ 分类器输出shape: {logits.shape}")

        # 5. 测试内存占用
        print("\n5️⃣  测试内存占用...")
        model_size = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"   - 模型参数量: {model_size:.2f}M")

        if model_size > 1000:
            print(f"   ⚠️  警告: 模型较大 ({model_size:.2f}M)，建议使用GPU或减小batch_size")

        print(f"\n{'='*60}")
        print("✅ 兼容性检查通过！可以安全使用此模型")
        print(f"{'='*60}\n")

        return True, {
            'hidden_size': model.config.hidden_size,
            'vocab_size': tokenizer.vocab_size,
            'has_pooler': has_pooler,
            'model_size': model_size
        }

    except Exception as e:
        print(f"\n❌ 错误: {str(e)}")
        print(f"\n可能的原因:")
        print(f"1. 模型名称错误")
        print(f"2. 缺少依赖: pip install transformers_stream_generator")
        print(f"3. 网络连接问题")
        print(f"4. 需要更新transformers版本: pip install --upgrade transformers")
        return False, None

def recommend_config(model_name, model_info):
    """根据模型信息推荐配置"""
    print("\n📋 推荐的config.py配置:")
    print("="*60)

    # 根据模型大小推荐batch_size
    if model_info['model_size'] > 1000:
        batch_size = 4
    elif model_info['model_size'] > 500:
        batch_size = 8
    else:
        batch_size = 16

    # 根据模型推荐序列长度
    if 'qwen' in model_name.lower():
        max_seq_length = 512  # Qwen支持更长序列
    else:
        max_seq_length = 128

    config_template = f'''
class Config:
    """
    模型配置类，包含所有可配置参数
    """
    # 模型参数
    model_name = "{model_name}"
    max_seq_length = {max_seq_length}  # Qwen支持更长序列
    num_classes = 2

    # 训练参数
    batch_size = {batch_size}  # 根据模型大小调整
    learning_rate = 2e-5
    num_epochs = 5

    # 路径配置
    train_path = "train.csv"
    dev_path = "dev.csv"
    test_path = "test.csv"
    model_save_path = "sentiment_model.pth"
'''

    print(config_template)
    print("="*60)

    print("\n⚠️  重要提示:")
    print(f"1. 需要在 model.py 和 main.py 中添加 trust_remote_code=True 参数")
    print(f"2. 建议batch_size={batch_size}（根据模型大小{model_info['model_size']:.0f}M调整）")
    print(f"3. 序列长度可以设为{max_seq_length}（Qwen支持更长序列）")
    if not model_info['has_pooler']:
        print(f"4. ✅ Qwen没有pooler_output，但代码已自动处理")

if __name__ == "__main__":
    print("🔍 Qwen模型兼容性检查工具\n")

    # 常用的Qwen模型列表
    qwen_models = [
        "Qwen/Qwen-1_8B",      # 1.8B参数，较小
        "Qwen/Qwen-7B",        # 7B参数，中等
        "Qwen/Qwen-14B",       # 14B参数，较大
        "Qwen/Qwen2-1.5B",     # Qwen2系列
        "Qwen/Qwen2-7B",
    ]

    print("可用的Qwen模型:")
    for i, model in enumerate(qwen_models, 1):
        print(f"  {i}. {model}")

    print("\n请选择要测试的模型 (输入数字)，或输入自定义模型名称:")
    user_input = input("> ").strip()

    # 判断是数字还是模型名称
    if user_input.isdigit() and 1 <= int(user_input) <= len(qwen_models):
        model_name = qwen_models[int(user_input) - 1]
    else:
        model_name = user_input

    # 运行测试
    success, model_info = test_qwen_model(model_name)

    if success:
        recommend_config(model_name, model_info)

        print("\n📝 下一步:")
        print("1. 修改 config.py 中的 model_name")
        print("2. 修改 model.py 和 main.py 添加 trust_remote_code=True")
        print("3. 运行训练: python main.py")
    else:
        print("\n❌ 兼容性检查失败，请解决上述问题后重试")
