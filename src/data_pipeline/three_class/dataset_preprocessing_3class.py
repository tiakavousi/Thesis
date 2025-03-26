import pandas as pd
import re
import emoji
import sys
import os
from pathlib import Path

# Add project root to import CONFIG_3CLASS
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.config_three_class import CONFIG_3CLASS

def clean_text(text):
    """
    Clean and preprocess the input text for transformer models with minimum cleaning.
    """
    text = str(text).strip()
    text = re.sub(r'http\S+|www\.\S+', '', text)      # Remove URLs
    text = re.sub(r'<.*?>', '', text)                 # Remove HTML tags
    text = emoji.demojize(text)                       # Convert emojis to text
    text = re.sub(r'\s+', ' ', text)                  # Normalize whitespace
    return text.lower()

def main():
    raw_dataset_path = CONFIG_3CLASS["dataset"]["raw_combined_dataset"]
    output_path = CONFIG_3CLASS["dataset"]["dataset_path"]
    random_seed = CONFIG_3CLASS["dataset"]["random_seed"]
    class_names = CONFIG_3CLASS["dataset"]["class_names"]

    print(f"📥 Loading dataset from: {raw_dataset_path}")
    df = pd.read_csv(raw_dataset_path)
    df.rename(columns={"text": "message"}, inplace=True)

    # Drop empty/NaN rows
    df = df.dropna(subset=["message"])
    df = df[df["message"].str.strip() != ""]

    # Clean text
    df["message"] = df["message"].apply(clean_text)
    df = df[df["message"].str.strip() != ""]

    # Filter to only keep valid sentiment classes (-1, 0, 1)
    valid_labels = [-1, 0, 1]
    df = df[df["sentiment"].isin(valid_labels)]

    # Shuffle entire dataset (but don't downsample)
    df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)

    # Save
    os.makedirs(Path(output_path).parent, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ Full (unbalanced) dataset saved to: {output_path}")

if __name__ == "__main__":
    main()
