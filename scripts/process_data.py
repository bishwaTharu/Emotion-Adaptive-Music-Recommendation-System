import pandas as pd
import numpy as np
import yaml

with open("config.yml", "r") as f:
    config = yaml.safe_load(f)

df = pd.read_csv(config["raw_data_path"])

# Assuming columns: id, split, V, A, D, text
# Add wide_feat as list of 7 zeros
df['wide_feat'] = [np.zeros(7).tolist()] * len(df)
train_df = df[df['split'] == 'train']
val_df = df[df['split'] == 'dev']  
train_df.to_parquet(config["data_train_path"])
val_df.to_parquet(config["data_val_path"])