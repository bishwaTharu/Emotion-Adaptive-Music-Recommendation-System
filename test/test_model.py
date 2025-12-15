import torch
from src.models.va_regressor import build_model
from transformers import AutoTokenizer

model = build_model()
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

text = "This is a test sentence."
encoding = tokenizer(text, truncation=True, padding="max_length", max_length=128, return_tensors="pt")
wide = torch.zeros(1, 7)

output = model(encoding["input_ids"], encoding["attention_mask"], wide)
print("Model output shape:", output.shape)
print("Output:", output)