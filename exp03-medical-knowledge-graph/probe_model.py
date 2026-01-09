from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
import torch

model_name = "d4data/biomedical-ner-all"
text = "Formate assay in body fluids: application in methanol poisoning. Patient had severe headache and nausea after taking aspirin."

try:
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    
    ner = pipeline("token-classification", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
    results = ner(text)
    
    print("\nResults (aggregation='simple'):")
    for r in results:
        print(r)

    # formatted
    print("\nFormatted:")
    for r in results:
        word = r['word']
        group = r['entity_group']
        print(f"{word} -> {group}")

except Exception as e:
    print(f"Error: {e}")
