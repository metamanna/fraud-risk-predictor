from transformers import pipeline
sentiment_pipeline = pipeline("sentiment-analysis")

def get_sentiment_score(text):
    result = sentiment_pipeline(text)[0]
    label = result["label"]
    score = result["score"]
    return score if "POSITIVE" in label else -score