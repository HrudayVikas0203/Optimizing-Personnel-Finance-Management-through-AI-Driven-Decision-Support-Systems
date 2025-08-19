from transformers import pipeline, GPT2Tokenizer, GPT2LMHeadModel
import torch

def load_gpt2_model():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    return tokenizer, model

def generate_summary(text, max_length=100):
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    summary = summarizer(text, max_length=max_length, min_length=30, do_sample=False)
    return summary[0]['summary_text']

def analyze_sentiment(text):
    sentiment_analyzer = pipeline("sentiment-analysis")
    result = sentiment_analyzer(text)
    return result[0]