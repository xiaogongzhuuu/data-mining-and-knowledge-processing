"""
RAG 系统自动化诊断脚本
检测所有系统组件的状态
"""

import os
import sys
import json
import requests

# 设置环境变量
os.environ['HF_HOME'] = './hf_cache'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'

def print_section(title):
    """打印分节标题"""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60 + "\n")

def check_environment():
    """检查环境"""
    print("1. Environment Check")
    print("-" * 40)

    # Python 版本
    print(f"   Python version: {sys.version.split()[0]}")
    if sys.version_info >= (3, 10):
        print("   [OK] Python version is compatible")
    else:
        print("   [WARNING] Python version should be 3.10+")

    # 工作目录
    print(f"   Working directory: {os.getcwd()}")
    expected_dir = "D:\\data-mining-and-knowledge-processing\\2025-spring\\exp04-easy-rag-system"
    if os.getcwd().replace("/", "\\") == expected_dir:
        print("   [OK] Working directory is correct")
    else:
        print(f"   [WARNING] Expected directory: {expected_dir}")

    # 环境变量
    print(f"   HF_HOME: {os.environ.get('HF_HOME', 'Not set')}")
    print(f"   TRANSFORMERS_OFFLINE: {os.environ.get('TRANSFORMERS_OFFLINE', 'Not set')}")

    return True

def check_dependencies():
    """检查依赖包"""
    print("\n2. Dependencies Check")
    print("-" * 40)

    required_packages = {
        'streamlit': 'streamlit',
        'sentence_transformers': 'sentence-transformers',
        'chromadb': 'chromadb',
        'requests': 'requests',
        'torch': 'torch'
    }

    all_ok = True
    for module_name, package_name in required_packages.items():
        try:
            module = __import__(module_name)
            version = getattr(module, '__version__', 'unknown')
            print(f"   [OK] {package_name}: {version}")
        except ImportError:
            print(f"   [FAILED] {package_name}: Not installed")
            all_ok = False

    return all_ok

def check_embedding_model():
    """检查嵌入模型"""
    print("\n3. Embedding Model Check")
    print("-" * 40)

    try:
        from sentence_transformers import SentenceTransformer

        # 检查模型路径
        cache_path = os.path.abspath('./hf_cache')
        snapshot_path = os.path.join(
            cache_path,
            'hub',
            'models--moka-ai--m3e-base',
            'snapshots',
            '764b537a0e50e5c7d64db883f2d2e051cbe3c64c'
        )

        if os.path.exists(snapshot_path):
            print(f"   [OK] Model cache exists: {snapshot_path}")

            # 检查关键文件
            model_file = os.path.join(snapshot_path, 'model.safetensors')
            if os.path.exists(model_file):
                size_mb = os.path.getsize(model_file) / (1024 * 1024)
                print(f"   [OK] Model file: {size_mb:.2f} MB")
            else:
                print(f"   [FAILED] Model file not found")
                return False

            # 尝试加载模型
            print("   Loading model...")
            model = SentenceTransformer(snapshot_path, device='cpu')
            print(f"   [OK] Model loaded successfully")
            print(f"   Model dimension: {model.get_sentence_embedding_dimension()}")

            # 测试编码
            test_text = "Test text"
            embedding = model.encode(test_text)
            print(f"   [OK] Encoding test passed")
            print(f"   Embedding shape: {embedding.shape}")

            return True
        else:
            print(f"   [FAILED] Model cache not found")
            return False

    except Exception as e:
        print(f"   [FAILED] Error: {e}")
        return False

def check_ollama():
    """检查 Ollama"""
    print("\n4. Ollama Check")
    print("-" * 40)

    try:
        # 检查 Ollama 服务
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            print("   [OK] Ollama service is running")

            # 检查模型列表
            models = response.json().get('models', [])
            print(f"   Available models: {len(models)}")

            # 检查 qwen3:8b
            qwen_available = any('qwen3:8b' in model.get('name', '') for model in models)
            if qwen_available:
                print("   [OK] qwen3:8b model is available")
            else:
                print("   [WARNING] qwen3:8b model not found")
                print("   Run: ollama pull qwen3:8b")

            return True
        else:
            print(f"   [FAILED] Ollama service returned status {response.status_code}")
            return False

    except requests.exceptions.ConnectionError:
        print("   [FAILED] Cannot connect to Ollama service")
        print("   Please run: ollama serve")
        return False
    except Exception as e:
        print(f"   [FAILED] Error: {e}")
        return False

def check_chromadb():
    """检查 ChromaDB"""
    print("\n5. ChromaDB Check")
    print("-" * 40)

    try:
        import chromadb

        # 检查数据库文件
        db_file = "chroma.sqlite3"
        if os.path.exists(db_file):
            print(f"   [OK] Database file exists: {db_file}")
        else:
            print(f"   [WARNING] Database file not found")
            print("   Database will be created on first run")

        # 尝试连接
        client = chromadb.PersistentClient(path=".")
        print("   [OK] ChromaDB client initialized")

        # 检查 collections
        collections = client.list_collections()
        print(f"   Collections: {len(collections)}")

        if collections:
            for coll in collections:
                count = coll.count()
                print(f"   - {coll.name}: {count} entities")

        return True

    except Exception as e:
        print(f"   [FAILED] Error: {e}")
        return False

def check_data():
    """检查数据文件"""
    print("\n6. Data Check")
    print("-" * 40)

    data_file = "./data/processed_data.json"

    if os.path.exists(data_file):
        print(f"   [OK] Data file exists: {data_file}")

        try:
            with open(data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            print(f"   [OK] Data loaded successfully")
            print(f"   Total documents: {len(data)}")

            if len(data) > 0:
                first_doc = data[0]
                print(f"   Sample document keys: {list(first_doc.keys())}")
            else:
                print("   [WARNING] No documents found")

            return True

        except Exception as e:
            print(f"   [FAILED] Error loading data: {e}")
            return False
    else:
        print(f"   [FAILED] Data file not found: {data_file}")
        return False

def run_diagnostics():
    """运行所有诊断"""
    print("="*60)
    print("  RAG System Diagnostics")
    print("="*60)

    results = {
        'Environment': check_environment(),
        'Dependencies': check_dependencies(),
        'Embedding Model': check_embedding_model(),
        'Ollama': check_ollama(),
        'ChromaDB': check_chromadb(),
        'Data': check_data()
    }

    # 汇总结果
    print_section("Diagnostics Summary")

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for name, result in results.items():
        status = "[OK]" if result else "[FAILED]"
        print(f"{status} {name}")

    print(f"\nTotal: {passed}/{total} checks passed")

    if passed == total:
        print("\n[SUCCESS] All checks passed! System is ready to use.")
        print("\nNext steps:")
        print("1. Start Ollama service: ollama serve")
        print("2. Run the system: streamlit run app.py")
        return 0
    else:
        print("\n[FAILED] Some checks failed. Please review the errors above.")
        print("\nFor troubleshooting, see: TROUBLESHOOTING.md")
        return 1

if __name__ == "__main__":
    exit_code = run_diagnostics()
    sys.exit(exit_code)
