import torch
from transformers import AutoTokenizer

# Load TinyBERT tokenizer
tokenizer = AutoTokenizer.from_pretrained('huawei-noah/TinyBERT_General_4L_312D')

def tokenize_texts(texts, max_length=128):
    encodings = tokenizer(
        list(texts),
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors='pt'
    )
    return encodings['input_ids'], encodings['attention_mask']
