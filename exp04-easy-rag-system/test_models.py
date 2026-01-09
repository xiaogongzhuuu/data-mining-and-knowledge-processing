"""
模型加载测试脚本
用于诊断和验证嵌入模型是否能正确加载
"""

import os
import sys

# 设置环境变量
os.environ['HF_HOME'] = './hf_cache'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFASET'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'

from sentence_transformers import SentenceTransformer

def test_model_loading():
    """测试模型加载"""
    print("="*60)
    print("模型加载测试")
    print("="*60)

    # 检查缓存路径
    cache_path = os.path.abspath('./hf_cache')
    print(f"\n1. 检查缓存路径: {cache_path}")
    print(f"   路径存在: {os.path.exists(cache_path)}")

    # 检查模型快照
    snapshot_path = os.path.join(
        cache_path,
        'hub',
        'models--moka-ai--m3e-base',
        'snapshots',
        '764b537a0e50e5c7d64db883f2d2e051cbe3c64c'
    )

    print(f"\n2. 检查模型快照: {snapshot_path}")
    print(f"   路径存在: {os.path.exists(snapshot_path)}")

    if os.path.exists(snapshot_path):
        # 列出模型文件
        files = os.listdir(snapshot_path)
        print(f"   模型文件: {len(files)} 个")
        for f in files[:10]:  # 只显示前10个
            file_path = os.path.join(snapshot_path, f)
            if os.path.isfile(file_path):
                size_mb = os.path.getsize(file_path) / (1024 * 1024)
                print(f"     - {f}: {size_mb:.2f} MB")

    # 尝试加载模型
    print(f"\n3. 尝试加载模型...")
    try:
        if os.path.exists(snapshot_path):
            print(f"   使用本地路径加载...")
            model = SentenceTransformer(snapshot_path, device='cpu')
        else:
            print(f"   使用模型名称加载...")
            model = SentenceTransformer('moka-ai/m3e-base', cache_folder=cache_path, device='cpu')

        print(f"   [OK] Model loaded successfully!")
        print(f"   Model dimension: {model.get_sentence_embedding_dimension()}")

        # 测试编码
        print(f"\n4. Testing encoding...")
        test_text = "这是一个测试文本"
        embedding = model.encode(test_text)
        print(f"   [OK] Encoding successful!")
        print(f"   Embedding dimension: {embedding.shape}")
        print(f"   First 5 values: {embedding[:5]}")

        return True

    except Exception as e:
        print(f"   [FAILED] Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_model_loading()
    print("\n" + "="*60)
    if success:
        print("[SUCCESS] All tests passed!")
    else:
        print("[FAILED] Tests failed, please check error messages")
    print("="*60)