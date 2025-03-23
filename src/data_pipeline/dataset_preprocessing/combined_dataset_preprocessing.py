import pandas as pd
import re
import emoji
import sys
import os
from pathlib import Path

# Add the project root to the path to import CONFIG
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from src.config import CONFIG

def clean_text(text):
    """
    Clean and preprocess the input text for transformer models with minimum cleaning.
    """
    # Ensure the text is a string and remove leading/trailing whitespace
    text = str(text).strip()
    
    # Remove URLs (patterns starting with http, https, or www)
    text = re.sub(r'http\S+|www\.\S+', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Convert emojis to text
    text = emoji.demojize(text)
    
    # Normalize multiple whitespace into one space
    text = re.sub(r'\s+', ' ', text)
    
    # Convert text to lower case
    text = text.lower()
    
    return text

def main():    
    # 1) Load the dataset from Excel
    raw_dataset_path = CONFIG["dataset"]["raw_combined_dataset"]
    print(f"Loading dataset from: {raw_dataset_path}")
    df = pd.read_excel(raw_dataset_path)
    
    # 2) Remove rows where 'message' is empty or NaN
    df = df.dropna(subset=["message"])                 # Remove rows with NaN in 'message'
    df = df[df["message"].str.strip() != ""]           # Remove rows where 'message' is just whitespace
    
    # 3) Clean the text in the 'message' column
    df["message"] = df["message"].apply(lambda x: clean_text(x))
    
    # Remove rows that might have become empty after cleaning
    df = df[df["message"].str.strip() != ""]
    
    # 4) Ensure 'sentiment' column contains only valid values (-1, 0)
    df = df[df["sentiment"].isin([-1, 0])]
    
    # 5) Find the minimum class count (for downsampling)
    class_counts = df["sentiment"].value_counts()
    min_count = class_counts[-1]  # Downsample to the count of negative sentiment (-1)


    # 6) Shuffle the dataset before downsampling to ensure randomness and avoiding any order bias.
    df = df.sample(frac=1, random_state=CONFIG["dataset"]["random_seed"]).reset_index(drop=True)
    
   # 7) Downsample each class to the min_count
    balanced_df = (
        df.groupby("sentiment")
        .apply(lambda x: x.sample(n=min_count, random_state=CONFIG["dataset"]["random_seed"]))
        .reset_index(drop=True)
    )
    
    # 8) Shuffle the dataset
    balanced_df = balanced_df.sample(frac=1, random_state=CONFIG["dataset"]["random_seed"]).reset_index(drop=True)
    
    # 9) Save the balanced dataset to a CSV file
    output_path = CONFIG["dataset"]["dataset_path"]
    balanced_df.to_csv(output_path, index=False)
    print(f"Balanced dataset saved to: {output_path}")

if __name__ == "__main__":
    main()
