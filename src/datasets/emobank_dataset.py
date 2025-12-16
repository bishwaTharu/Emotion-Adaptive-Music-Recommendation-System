import torch
import numpy as np
from torch.utils.data import Dataset
###
class EmoBankDataset(Dataset):
    """
    Dataset class for EmoBank.
    Target: Valence (V) and Arousal (A).
    Inputs:
        - Deep: Tokenized text.
        - Wide: One-hot encoded sentence length bucket.
    """
    def __init__(self, dataframe, tokenizer, max_length, length_buckets):
        self.df = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.buckets = length_buckets
        word_counts = self.df['text'].apply(lambda x: len(str(x).split())).values
        bucket_indices = np.digitize(word_counts, self.buckets) - 1
        bucket_indices = np.clip(bucket_indices, 0, 6)
        self.wide_features = np.eye(7)[bucket_indices].astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = str(row['text'])
        target = torch.tensor([row['V'], row['A']], dtype=torch.float32)
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        wide_feat = torch.tensor(self.wide_features[idx], dtype=torch.float32)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'wide_features': wide_feat,
            'target': target
        }

# this is a test