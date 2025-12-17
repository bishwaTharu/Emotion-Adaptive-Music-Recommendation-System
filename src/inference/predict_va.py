import torch
import joblib
from typing import Any

def predict_va(text: str, model: torch.nn.Module, tokenizer: Any, device: torch.device) -> torch.Tensor:
    model.eval()

    encoding = tokenizer(
        text=text,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt"
    )

    wide = torch.zeros(1, 7)  # neutral wide feature at inference

    with torch.no_grad():
        va_scaled = model(
            encoding["input_ids"].to(device),
            encoding["attention_mask"].to(device),
            wide.to(device)
        )

    # Inverse standardization
    va = va_scaled.cpu().numpy()[0]
    return torch.tensor(va, dtype=torch.float32)
