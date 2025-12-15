import pandas as pd
import numpy as np

df = pd.read_csv('data/raw/emobank.csv')

# Assuming columns: id, split, V, A, D, text
# Add wide_feat as list of 7 zeros
df['wide_feat'] = [np.zeros(7).tolist()] * len(df)
train_df = df[df['split'] == 'train']
val_df = df[df['split'] == 'dev']  
train_df.to_parquet('data/processed/emobank_train.parquet')
val_df.to_parquet('data/processed/emobank_val.parquet')