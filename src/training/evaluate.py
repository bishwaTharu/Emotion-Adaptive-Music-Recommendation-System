import torch
from ..utils.metrics import r2_score, pearson_corr

def evaluate(model, loader, device):
    model.eval()
    preds, targets = [], []

    with torch.no_grad():
        for batch in loader:
            p = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch["wide_features"].to(device),
            )
            preds.append(p)
            targets.append(batch["target"].to(device))

    preds = torch.cat(preds)
    targets = torch.cat(targets)

    return {
        "val_r2": r2_score(targets, preds).item(),
        "val_pearson": pearson_corr(targets, preds).item(),
    }
