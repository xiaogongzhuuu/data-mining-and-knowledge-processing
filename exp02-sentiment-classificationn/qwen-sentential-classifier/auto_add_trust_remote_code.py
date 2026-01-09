#!/usr/bin/env python3
"""
自动为Qwen模型添加trust_remote_code参数
此脚本会修改所有需要的文件，添加 trust_remote_code=True 参数
"""

import os
import re
from pathlib import Path

def backup_file(file_path):
    """备份文件"""
    backup_path = f"{file_path}.backup"
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    with open(backup_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"   ✅ 备份创建: {backup_path}")

def add_trust_remote_code(file_path):
    """在from_pretrained调用中添加trust_remote_code=True"""

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    original_content = content
    modified = False

    # 模式1: AutoTokenizer.from_pretrained(xxx)
    pattern1 = r'(AutoTokenizer\.from_pretrained\s*\(\s*[^,\)]+)(\s*\))'

    def replace1(match):
        nonlocal modified
        # 检查是否已经有trust_remote_code
        if 'trust_remote_code' in match.group(0):
            return match.group(0)
        modified = True
        return match.group(1) + ',\n        trust_remote_code=True' + match.group(2)

    content = re.sub(pattern1, replace1, content)

    # 模式2: AutoModel.from_pretrained(xxx)
    pattern2 = r'(AutoModel\.from_pretrained\s*\(\s*[^,\)]+)(\s*\))'

    def replace2(match):
        nonlocal modified
        if 'trust_remote_code' in match.group(0):
            return match.group(0)
        modified = True
        return match.group(1) + ',\n        trust_remote_code=True' + match.group(2)

    content = re.sub(pattern2, replace2, content)

    # 模式3: 多行from_pretrained
    # 处理已经有参数的情况
    pattern3 = r'((?:AutoTokenizer|AutoModel)\.from_pretrained\s*\([^)]*?model_name[^)]*?)(\))'

    def replace3(match):
        nonlocal modified
        full_match = match.group(0)
        if 'trust_remote_code' in full_match:
            return full_match
        # 在最后一个参数后添加
        modified = True
        return match.group(1).rstrip() + ',\n        trust_remote_code=True' + match.group(2)

    content = re.sub(pattern3, replace3, content, flags=re.DOTALL)

    if modified:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    return False

def main():
    print("🔧 Qwen模型自动配置工具")
    print("="*60)
    print("此脚本将自动在所有必要的文件中添加 trust_remote_code=True 参数\n")

    # 获取当前目录
    current_dir = Path(__file__).parent

    # 需要修改的文件列表
    files_to_modify = [
        'model.py',
        'main.py',
        'compare_trained_untrained.py',
        'sample_stability_analysis.py',
        'train_size_analysis.py',
        'epoch_analysis.py',
    ]

    print("将要修改的文件:")
    for f in files_to_modify:
        print(f"  - {f}")

    print("\n⚠️  重要提示:")
    print("1. 脚本会自动创建 .backup 备份文件")
    print("2. 如果出现问题，可以从备份恢复")
    print("3. 建议先在测试环境运行\n")

    response = input("是否继续? (y/n): ").strip().lower()
    if response != 'y':
        print("❌ 操作已取消")
        return

    print("\n开始修改...\n")

    modified_count = 0
    for file_name in files_to_modify:
        file_path = current_dir / file_name

        if not file_path.exists():
            print(f"⏭️  跳过 {file_name} (文件不存在)")
            continue

        print(f"📝 处理 {file_name}...")

        # 创建备份
        backup_file(file_path)

        # 修改文件
        was_modified = add_trust_remote_code(file_path)

        if was_modified:
            print(f"   ✅ 已添加 trust_remote_code=True")
            modified_count += 1
        else:
            print(f"   ℹ️  无需修改（已包含或无相关代码）")

        print()

    print("="*60)
    print(f"✅ 完成！共修改了 {modified_count} 个文件\n")

    if modified_count > 0:
        print("📋 下一步:")
        print("1. 修改 config.py 中的 model_name")
        print("2. 运行: python check_qwen_compatibility.py")
        print("3. 测试训练: python main.py")
        print("\n如果遇到问题，可以从 .backup 文件恢复")
    else:
        print("ℹ️  所有文件都已正确配置或无需修改")

    print("\n🗑️  清理备份文件:")
    cleanup = input("是否删除所有 .backup 文件? (y/n): ").strip().lower()
    if cleanup == 'y':
        for backup in current_dir.glob("*.backup"):
            backup.unlink()
            print(f"   删除: {backup.name}")
        print("✅ 备份文件已清理")

if __name__ == "__main__":
    main()
