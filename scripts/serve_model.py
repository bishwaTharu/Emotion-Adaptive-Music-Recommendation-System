import sys
sys.path.append('..')

import torch
import yaml
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.models.va_regressor import build_model
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
    model.load_state_dict(torch.load(config["model_checkpoint_path"], map_location=device))
    logger.info("Model checkpoint loaded successfully.")
except FileNotFoundError:
    logger.warning("No checkpoint found. Please train the model first.")
    raise RuntimeError("Model checkpoint not found.")

model.eval()
tokenizer = AutoTokenizer.from_pretrained(config["transformer"])
logger.info("Tokenizer loaded successfully.")

app = FastAPI(title="Emotion-Aware Music Recommender API", version="1.0.0")

class InferenceRequest(BaseModel):
    text: str

class InferenceResponse(BaseModel):
    valence: float
    arousal: float

@app.post("/infer", response_model=InferenceResponse)
def infer_emotion(request: InferenceRequest):
    try:
        encoding = tokenizer(request.text, truncation=True, padding="max_length", max_length=config["max_length"], return_tensors="pt")
        wide = torch.zeros(1, 7).to(device)

        with torch.no_grad():
            output = model(encoding["input_ids"].to(device), encoding["attention_mask"].to(device), wide)

        va = output.squeeze().tolist()
        logger.info(f"Inferred VA for text: Valence={va[0]:.3f}, Arousal={va[1]:.3f}")
        return InferenceResponse(valence=round(va[0], 3), arousal=round(va[1], 3))
    except Exception as e:
        logger.error(f"Error during inference: {str(e)}")
        raise HTTPException(status_code=500, detail="Inference failed")

@app.get("/")
def root():
    return {"message": "Emotion-Aware Music Recommender API is running"}

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting FastAPI server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)