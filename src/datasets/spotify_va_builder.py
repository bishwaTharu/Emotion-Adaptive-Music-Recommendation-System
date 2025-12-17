import torch
import pandas as pd
from tqdm import tqdm

def build_spotify_va(
    lyrics_df,
    model,
    tokenizer,
    device,
    chunk_size=128
):
    records = []

    model.eval()

    for _, row in tqdm(lyrics_df.iterrows(), total=len(lyrics_df)):
        lyrics = row["lyrics"]
        chunks = [
            lyrics[i:i+chunk_size]
            for i in range(0, len(lyrics), chunk_size)
        ]

        va_preds = []

        for chunk in chunks:
            encoding = tokenizer(
                text=chunk,
                truncation=True,
                padding="max_length",
                max_length=128,
                return_tensors="pt"
            )

            wide = torch.zeros(1, 7)

            with torch.no_grad():
                va_scaled = model(
                    encoding["input_ids"].to(device),
                    encoding["attention_mask"].to(device),
                    wide.to(device)
                )

            va = va_scaled.cpu().numpy()[0]
            va_preds.append(va)

        mean_va = sum(va_preds) / len(va_preds)

        records.append({
            "song": row["song"],
            "artist": row["artist"],
            "V": mean_va[0],
            "A": mean_va[1],
        })

    return pd.DataFrame(records)
