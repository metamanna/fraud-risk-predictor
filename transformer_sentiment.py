# modules/transformer_sentiment.py

from transformers import pipeline, AutoTokenizer, AutoModel
import torch
import numpy as np

# ✅ Load sentiment pipeline (for score only)
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased")

def get_transformer_sentiment(text):
    result = sentiment_pipeline(text[:512])[0]
    label = result['label']
    score = result['score']
    return score if label == "POSITIVE" else -score

# ✅ Load model + tokenizer once (for embeddings)
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModel.from_pretrained("distilbert-base-uncased")
model.eval()

def get_text_embedding(text):
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token
        return embeddings.squeeze().numpy()  # shape: (768,)