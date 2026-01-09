import json
import os

try:
    with open('data/processed/processed_articles.json', 'r') as f:
        data = json.load(f)
        print(f"Total articles: {len(data)}")
        if data:
            print("First article sample:")
            art = data[0]
            print(json.dumps(art, indent=2, ensure_ascii=False))
            
            # Check for co-occurrence in first 10 articles
            print("\nEntity Co-occurrence Stats (first 10):")
            for i, art in enumerate(data[:10]):
                counts = {k: len(v) for k, v in art.items() if isinstance(v, list)}
                print(f"Art {i}: {counts}")

except Exception as e:
    print(f"Error reading file: {e}")
