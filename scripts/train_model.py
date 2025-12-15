import torch
import torch.nn as nn
import mlflow
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models.va_regressor import build_model
from src.data.emobank_dataset import EmoBankDataset
from src.training.train import train_epoch
from src.training.evaluate import evaluate
from src.utils.logging import setup_mlflow, log_params, log_metrics, log_model
from src.utils.seed import set_seed
import pandas as pd
import yaml
import matplotlib.pyplot as plt


def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --------------------
    # MLflow setup
    # --------------------
    setup_mlflow(experiment_name="words_to_waves_va_regression")

    with mlflow.start_run():
        # --------------------
        # Config
        # --------------------
        with open("config.yml", "r") as f:
            config = yaml.safe_load(f)

        log_params(config)

        # --------------------
        # Data
        # --------------------
        train_df = pd.read_parquet("data/processed/emobank_train.parquet")
        val_df = pd.read_parquet("data/processed/emobank_val.parquet")

        tokenizer = BertTokenizer.from_pretrained(config["transformer"])

        train_dataset = EmoBankDataset(train_df, tokenizer, config["max_length"], config["length_buckets"])
        val_dataset = EmoBankDataset(val_df, tokenizer, config["max_length"], config["length_buckets"])

        train_loader = DataLoader(
            train_dataset,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=0 if device.type == "cpu" else 4,
            pin_memory=device.type != "cpu",
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=0 if device.type == "cpu" else 4,
            pin_memory=device.type != "cpu",
        )

        # --------------------
        # Model
        # --------------------
        model = build_model(config["transformer"])
        model.to(device)

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config["lr"]
        )

        total_steps = len(train_loader) * config["epochs"]
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps,
        )

        # --------------------
        # Training loop
        # --------------------
        train_losses = []
        val_losses = []
        for epoch in range(config["epochs"]):
            train_loss = train_epoch(
                model,
                train_loader,
                optimizer,
                scheduler,
                device,
            )
            train_losses.append(train_loss)

            val_metrics = evaluate(model, val_loader, device)
            val_loss = val_metrics.get('val_loss', 0)  # Assuming evaluate returns val_loss, but it doesn't, wait.
            model.eval()
            val_loss = 0.0

            with torch.no_grad():

                for batch in val_loader:
                    preds = model(batch["input_ids"].to(device), batch["attention_mask"].to(device), batch["wide_features"].to(device))
                    loss = nn.SmoothL1Loss()(preds, batch["target"].to(device))
                    val_loss += loss.item()

            val_loss /= len(val_loader)
            val_losses.append(val_loss)
            log_metrics(

                {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    **val_metrics,
                },
                step=epoch
            )
            print(
                f"Epoch {epoch+1} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"R2: {val_metrics['val_r2']:.4f} | "
                f"Pearson: {val_metrics['val_pearson']:.4f}"
            )

        log_model(model)
        torch.save(model.state_dict(), "model_checkpoint.pth")
        with open("config_saved.yml", "w") as f:
            yaml.dump(config, f)
        print("\nTraining Losses:", train_losses)
        print("Validation Losses:", val_losses)

        # Plot
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(train_losses)+1), train_losses, label='Training Loss')
        plt.plot(range(1, len(val_losses)+1), val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.savefig('loss_plot.png')
        plt.show()

if __name__ == "__main__":
    main()
