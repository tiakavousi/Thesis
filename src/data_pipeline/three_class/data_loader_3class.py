import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import logging
from pathlib import Path
from src.config_three_class import CONFIG_3CLASS as CONFIG 

logger = logging.getLogger(__name__)

class ReviewDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        
        encoding = self.tokenizer.encode_plus(
            text, 
            add_special_tokens=True, 
            max_length=self.max_length,
            truncation=True, 
            padding="max_length", 
            return_attention_mask=True, 
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(label, dtype=torch.long),
        }


class DataModule:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.config = CONFIG["training"]

    def load_dataset(self):
        df = pd.read_csv(CONFIG["dataset"]["dataset_path"])
        df = df.dropna(subset=["message"])
        df = df.drop_duplicates(subset=["message"])

        texts = df["message"].astype(str).tolist()

        # Map sentiment: -1 → 0, 0 → 1, 1 → 2
        labels = [(l + 1) for l in df["sentiment"].astype(int).tolist()]

        logger.info(f"Dataset loaded with {len(texts)} samples.")
        return texts, labels

    def create_dataloaders(self, texts, labels):
        train_ratio = CONFIG["dataset"]["train_ratio"]
        val_ratio = CONFIG["dataset"]["val_ratio"]
        test_ratio = CONFIG["dataset"]["test_ratio"]
        random_seed = CONFIG["dataset"]["random_seed"]

        train_texts, temp_texts, train_labels, temp_labels = train_test_split(
            texts, labels, train_size=train_ratio, random_state=random_seed, stratify=labels
        )

        remaining_ratio = val_ratio / (val_ratio + test_ratio)
        val_texts, test_texts, val_labels, test_labels = train_test_split(
            temp_texts, temp_labels, train_size=remaining_ratio, 
            random_state=random_seed, stratify=temp_labels
        )

        batch_size = self.config["batch_size"]
        max_length = self.config["max_length"]

        train_dataset = ReviewDataset(train_texts, train_labels, self.tokenizer, max_length)
        val_dataset = ReviewDataset(val_texts, val_labels, self.tokenizer, max_length)
        test_dataset = ReviewDataset(test_texts, test_labels, self.tokenizer, max_length)

        return (
            DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
            DataLoader(val_dataset, batch_size=batch_size, shuffle=False),
            DataLoader(test_dataset, batch_size=batch_size, shuffle=False),
        )
