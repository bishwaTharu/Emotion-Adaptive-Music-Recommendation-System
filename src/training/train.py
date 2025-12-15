import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler

def train_epoch(
    model,
    loader,
    optimizer,
    scheduler,
    device,
):
    model.train()
    scaler = GradScaler()
    criterion = nn.SmoothL1Loss()
    total_loss = 0.0

    for batch in loader:
        optimizer.zero_grad()

        if device.type == "cuda":
            with autocast(device_type=device.type):
                preds = model(
                    batch["input_ids"].to(device),
                    batch["attention_mask"].to(device),
                    batch["wide_features"].to(device),
                )
                loss = criterion(preds, batch["target"].to(device))

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            preds = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch["wide_features"].to(device),
            )
            loss = criterion(preds, batch["target"].to(device))
            loss.backward()
            optimizer.step()

        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(loader)
