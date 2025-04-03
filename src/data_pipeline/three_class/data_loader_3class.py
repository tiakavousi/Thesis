import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from src.config_three_class import CONFIG_3CLASS as CONFIG 
from collections import Counter

class ReviewDataset(Dataset):
    """
    Custom PyTorch Dataset for tokenizing text data and preparing input tensors
    for transformer-based models.
    """
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        """
        Tokenizes the input text and returns a dictionary of input tensors.
        """
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        
        encoding = self.tokenizer.encode_plus(
            text, 
            add_special_tokens=True,  # Add special tokens like [CLS] and [SEP]
            max_length=self.max_length,  # Truncate/pad to max length
            truncation=True,
            padding="max_length",
            return_attention_mask=True,  # Include attention mask
            return_tensors="pt"  # Return as PyTorch tensors
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(label, dtype=torch.long),
        }


class DataModule:
    """
    Handles data loading, preprocessing, and splitting into train/val/test sets.
    """
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.config = CONFIG["training"]

    def load_dataset(self):
        """
        Loads and cleans the dataset, mapping sentiment labels from (-1, 0, 1) to (0, 1, 2).
        """
        df = pd.read_csv(CONFIG["dataset"]["dataset_path"])
        df = df.dropna(subset=["message"])  # Remove rows with null messages
        df = df.drop_duplicates(subset=["message"])  # Remove duplicates

        texts = df["message"].astype(str).tolist()

        # Convert sentiment labels from [-1, 0, 1] to [0, 1, 2] for model compatibility
        labels = [(l + 1) for l in df["sentiment"].astype(int).tolist()]

        print(f"Dataset loaded with {len(texts)} samples.")
        return texts, labels
    

    def get_class_weights(self):
        """
        Calculates normalized inverse-frequency weights for each sentiment class
        based on training label distribution.
        Returns a list of floats to be passed to CrossEntropyLoss.
        """
        texts, labels = self.load_dataset()

        # Count occurrences of each class (assumes labels are [0, 1, 2])
        label_counts = Counter(labels)
        total_samples = sum(label_counts.values())

        # Compute raw inverse frequency weights
        weights = [total_samples / label_counts[i] for i in range(len(label_counts))]

        # Normalize so weights sum to 1 (optional but makes comparison cleaner)
        weight_sum = sum(weights)
        normalized_weights = [w / weight_sum for w in weights]

        print(f"[INFO] Computed class weights: {normalized_weights}")
        return normalized_weights


    def create_dataloaders(self, texts, labels):
        """
        Splits the dataset into train, validation, and test sets using stratified sampling,
        then wraps them into DataLoader objects.
        """
        train_ratio = CONFIG["dataset"]["train_ratio"]
        val_ratio = CONFIG["dataset"]["val_ratio"]
        test_ratio = CONFIG["dataset"]["test_ratio"]
        random_seed = CONFIG["dataset"]["random_seed"]

        # First, split off the training set with stratification to preserve class distribution
        train_texts, temp_texts, train_labels, temp_labels = train_test_split(
            texts, labels, train_size=train_ratio, random_state=random_seed, stratify=labels
        )

        # Then split the remaining data into validation and test sets
        remaining_ratio = val_ratio / (val_ratio + test_ratio)
        val_texts, test_texts, val_labels, test_labels = train_test_split(
            temp_texts, temp_labels, train_size=remaining_ratio, 
            random_state=random_seed, stratify=temp_labels
        )

        batch_size = self.config["batch_size"]
        max_length = self.config["max_length"]
        
        # Create Dataset objects for each split
        train_dataset = ReviewDataset(train_texts, train_labels, self.tokenizer, max_length)
        val_dataset = ReviewDataset(val_texts, val_labels, self.tokenizer, max_length)
        test_dataset = ReviewDataset(test_texts, test_labels, self.tokenizer, max_length)

        # Return DataLoaders for training, validation, and testing
        return (
            DataLoader(train_dataset, batch_size=batch_size, shuffle=True),   # Shuffle for training
            DataLoader(val_dataset, batch_size=batch_size, shuffle=False),    # No shuffle for validation
            DataLoader(test_dataset, batch_size=batch_size, shuffle=False),   # No shuffle for test
        )
