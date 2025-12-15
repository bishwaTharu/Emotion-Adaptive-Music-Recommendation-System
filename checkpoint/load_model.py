import torch
import yaml
from src.models.va_regressor import build_model
from transformers import AutoTokenizer

# Load config
with open("config_saved.yml", "r") as f:
    config = yaml.safe_load(f)

# Build model
model = build_model(config["transformer"])

# Load checkpoint
model.load_state_dict(torch.load("model_checkpoint.pth"))
model.eval()

# Test
tokenizer = AutoTokenizer.from_pretrained(config["transformer"])
text = "I feel happy and excited."
encoding = tokenizer(text, truncation=True, padding="max_length", max_length=config["max_length"], return_tensors="pt")
wide = torch.zeros(1, 7)

with torch.no_grad():
    output = model(encoding["input_ids"], encoding["attention_mask"], wide)

print(f"Predicted V-A: {output.squeeze().tolist()}")