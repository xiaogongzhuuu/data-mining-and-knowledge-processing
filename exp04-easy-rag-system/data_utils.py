import json
import streamlit as st

def load_data(filepath):
    """从 JSON 文件加载数据。"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 将加载信息改为后台打印
        print(f"DEBUG: Successfully loaded {len(data)} articles from {filepath}")
        return data
    except FileNotFoundError:
        print(f"ERROR: Data file not found: {filepath}")
        return []
    except json.JSONDecodeError:
        print(f"ERROR: Error decoding JSON from file: {filepath}")
        return []
    except Exception as e:
        print(f"ERROR: An error occurred loading data: {e}")
        return []