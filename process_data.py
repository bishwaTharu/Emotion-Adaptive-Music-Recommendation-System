import pandas as pd
import numpy as np

# Load the raw CSV
df = pd.read_csv('data/raw/emobank.csv')

# Assuming columns: id, split, V, A, D, text
# Add wide_feat as list of 7 zeros
df['wide_feat'] = [np.zeros(7).tolist()] * len(df)

# Split into train and val
train_df = df[df['split'] == 'train']
val_df = df[df['split'] == 'dev']  # or 'test' if dev not present

# Save to parquet
train_df.to_parquet('data/processed/emobank_train.parquet')
val_df.to_parquet('data/processed/emobank_val.parquet')

print("Data processed successfully")