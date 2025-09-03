# data_loader.py

import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

def _read_text_file(filepath):
    """Reads content from a given file path."""
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    else:
        print(f"Warning: File not found at {filepath}")
        return ""

class TextPairDataset(Dataset):
    """
    A custom PyTorch Dataset for loading text pairs and their labels,
    formatted for BERT input.
    """
    def __init__(self, dataframe, tokenizer, max_len, data_dir, file1_name, file2_name):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.max_len = max_len
        self.data_dir = data_dir
        self.file1_name = file1_name
        self.file2_name = file2_name

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        sample_id = self.data.iloc[index]['id']
        real_text_id = self.data.iloc[index]['real_text_id']

        # Construct file paths dynamically based on sample_id
        article_dir = os.path.join(self.data_dir, f"article_{str(sample_id).zfill(4)}")
        file1_path = os.path.join(article_dir, self.file1_name)
        file2_path = os.path.join(article_dir, self.file2_name)
        text1_content = _read_text_file(file1_path)
        text2_content = _read_text_file(file2_path)

        if not text1_content or not text2_content:
            # Skip this sample by returning None, to be filtered out in DataLoader collate_fn
            return None

        # BERT expects inputs in a specific format for "sentence pair" tasks:
        # [CLS] text_a [SEP] text_b [SEP]
        encoding = self.tokenizer.encode_plus(
            text1_content,
            text2_content,
            add_special_tokens=True, # Add [CLS], [SEP]
            max_length=self.max_len,
            return_token_type_ids=True, # Differentiate between text_a and text_b
            padding='max_length',
            truncation=True,
            return_attention_mask=True, # Indicate actual tokens vs padding
            return_tensors='pt', # Return PyTorch tensors
        )

        # Label: 1 if file_1 is real, 0 if file_2 is real
        label = 1 if real_text_id == 1 else 0

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'token_type_ids': encoding['token_type_ids'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def create_data_loaders(train_csv_path, tokenizer, max_len, data_dir, file1_name, file2_name, batch_size, test_split=0.2):
    """
    Loads data from CSV, creates a TextPairDataset, and splits it into
    training and (optional) validation DataLoaders.
    """
    train_df = pd.read_csv(train_csv_path)

    dataset = TextPairDataset(train_df, tokenizer, max_len, data_dir, file1_name, file2_name)

    # Custom collate_fn to filter out None samples
    def collate_fn(batch):
        batch = [item for item in batch if item is not None]
        return torch.utils.data.default_collate(batch) if batch else None

    if len(dataset) < 2:
        print("\n--- WARNING: INSUFFICIENT DATA ---")
        print("The provided dataset contains only ONE training sample. Returning single DataLoaders.")
        train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        return train_dataloader, None # No separate validation set in this case
    else:
        train_size = int((1 - test_split) * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        print(f"Split data: {len(train_dataset)} training samples, {len(val_dataset)} validation samples.")
        return train_dataloader, val_dataloader