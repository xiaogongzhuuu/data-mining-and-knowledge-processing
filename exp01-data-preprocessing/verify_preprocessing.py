import re
from nltk.tokenize import word_tokenize
import nltk

nltk.download('punkt')

def preprocess_text(text):
    text = text.lower()
    # The fix: replace with space instead of empty string
    text = re.sub(r'[^\w\s]', ' ', text)
    tokens = word_tokenize(text)
    return tokens

text = "Great for the non-audiophile"
tokens = preprocess_text(text)
print(f"Original: {text}")
print(f"Tokens: {tokens}")

if "non" in tokens and "audiophile" in tokens:
    print("SUCCESS: Words are separated.")
else:
    print("FAILURE: Words are merged.")
