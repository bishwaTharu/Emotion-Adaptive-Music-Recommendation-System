import sys
sys.path.append('src')

import torch
import yaml
import logging
from models.va_regressor import build_model
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


with open("config.yml", "r") as f:
    config = yaml.safe_load(f)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

model = build_model(config["transformer"])
model.to(device)
try:
    model.load_state_dict(torch.load("model_checkpoint.pth", map_location=device))
    logger.info("Model checkpoint loaded successfully.")
except FileNotFoundError:
    logger.warning("No checkpoint found. Please train the model first.")

model.eval()
tokenizer = AutoTokenizer.from_pretrained(config["transformer"])

def infer_va(text):
    encoding = tokenizer(text, truncation=True, padding="max_length", max_length=config["max_length"], return_tensors="pt")
    wide = torch.zeros(1, 7).to(device)  # Assuming 7 wide features as in dataset

    with torch.no_grad():
        output = model(encoding["input_ids"].to(device), encoding["attention_mask"].to(device), wide)

    return output.squeeze().tolist()

if __name__ == "__main__":
    logger.info("Emotion-Aware Music Recommender Engine Started!")
    # Sample text
    text = "I feel happy and excited about the weekend!"
    va = infer_va(text)
    logger.info(f"Sample Text: '{text}'")
    logger.info(f"Predicted VA: Valence={va[0]:.3f}, Arousal={va[1]:.3f}")
    logger.info("Engine is ready for inference.")