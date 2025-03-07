import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    AdamW,
    get_linear_schedule_with_warmup,
)
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# =============================================================================
# Custom Dataset Class
# =============================================================================
class ReviewDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        """
        Initializes the dataset.

        Parameters:
        -----------
        texts : list of str
            The text data.
        labels : list of int
            The corresponding labels (e.g., 0 or 1).
        tokenizer : transformers.PreTrainedTokenizer
            The DistilBERT tokenizer.
        max_length : int, optional
            Maximum token length for the sequences.
        """
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
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),  # Tensor shape: (max_length)
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(label, dtype=torch.long),
        }


# =============================================================================
# Data Loading and Preparation Functions
# =============================================================================
def load_dataset(csv_file):
    """
    Load the dataset from CSV.

    Parameters:
    -----------
    csv_file : str
        Path to the CSV file.

    Returns:
    --------
    texts, labels : tuple of lists
    """
    df = pd.read_csv(csv_file)
    texts = df["message"].tolist()
    labels = df["is_toxic"].tolist()
    return texts, labels


def create_dataloaders(texts, labels, tokenizer, batch_size=16, max_length=128, test_size=0.1):
    """
    Split the dataset into training and validation sets and create DataLoaders.

    Parameters:
    -----------
    texts : list of str
        The text data.
    labels : list of int
        The corresponding labels.
    tokenizer : transformers.PreTrainedTokenizer
        The DistilBERT tokenizer.
    batch_size : int, optional
        The batch size.
    max_length : int, optional
        Maximum token length.
    test_size : float, optional
        Proportion of data to use for validation.

    Returns:
    --------
    train_loader, val_loader : DataLoader objects
    """
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=test_size, random_state=42
    )
    train_dataset = ReviewDataset(train_texts, train_labels, tokenizer, max_length)
    val_dataset = ReviewDataset(val_texts, val_labels, tokenizer, max_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


# =============================================================================
# Training Function
# =============================================================================
def train_model(model, train_loader, val_loader, epochs=3, lr=2e-5, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Fine-tune the model using the provided DataLoaders.

    Parameters:
    -----------
    model : torch.nn.Module
        The DistilBERT model.
    train_loader : DataLoader
        Training data loader.
    val_loader : DataLoader
        Validation data loader.
    epochs : int, optional
        Number of training epochs.
    lr : float, optional
        Learning rate.
    device : str, optional
        Device to run training on ("cuda" or "cpu").

    Returns:
    --------
    model : torch.nn.Module
        The fine-tuned model.
    """
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        model.train()
        total_train_loss = 0
        progress_bar = tqdm(train_loader, desc="Training", leave=False)
        for batch in progress_bar:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_train_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_train_loss = total_train_loss / len(train_loader)
        print(f"Average Training Loss: {avg_train_loss:.4f}")

        # Evaluation on validation set
        model.eval()
        total_eval_loss = 0
        correct = 0
        total = 0
        for batch in tqdm(val_loader, desc="Evaluating", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_eval_loss += loss.item()
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
        avg_val_loss = total_eval_loss / len(val_loader)
        accuracy = correct / total
        print(f"Validation Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.4f}")

    return model


# =============================================================================
# Main Function
# =============================================================================
def main():
    # Paths and hyperparameters
    csv_file = "/Users/tayebekavousi/Desktop/github_sa/datasets/preprocessed/code-review-dataset-balanced.csv"
    batch_size = 16
    max_length = 128
    epochs = 3
    lr = 2e-5
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Load the tokenizer and model (using the default DistilBERT tokenizer)
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

    # 2. Load dataset from CSV
    texts, labels = load_dataset(csv_file)

    # 3. Create DataLoaders for training and validation
    train_loader, val_loader = create_dataloaders(texts, labels, tokenizer, batch_size, max_length)

    # 4. Fine-tune the model
    model = train_model(model, train_loader, val_loader, epochs, lr, device)

    # 5. Save the fine-tuned model and tokenizer
    model_save_path = "/Users/tayebekavousi/Desktop/github_sa/models/distilbert_finetuned"
    os.makedirs(model_save_path, exist_ok=True)
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    print(f"Model and tokenizer saved to {model_save_path}")


if __name__ == "__main__":
    main()
