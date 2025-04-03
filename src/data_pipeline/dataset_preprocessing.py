import pandas as pd
import re
import emoji
import sys
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
    # 1) Load the dataset
    raw_dataset_path = CONFIG["dataset"]["raw_DeepSentimentSE_dataset"]
    print(f"Loading dataset from: {raw_dataset_path}")
    df = pd.read_csv(raw_dataset_path)
    
    # 2) Remove rows where 'text' is empty or NaN
    df = df.dropna(subset=["text"])                 # Remove rows with NaN in 'text'
    df = df[df["text"].str.strip() != ""]           # Remove rows where 'text' is just whitespace
    
    # 3) Clean the text in the 'text' column
    df["text"] = df["text"].apply(lambda x: clean_text(x))
    
    # Remove rows that might have become empty after cleaning
    df = df[df["text"].str.strip() != ""]
    
    # 4) Ensure 'sentiment' column contains only valid values (-1, 0)
    df = df[df["sentiment"].isin([-1, 0,1])]
    
    # 5) Shuffle the dataset
    df = df.sample(frac=1, random_state=CONFIG["dataset"]["random_seed"]).reset_index(drop=True)
    
    # 6) Save the full (not downsampled) cleaned dataset to a CSV file
    output_path = CONFIG["dataset"]["dataset_path"]
    df.to_csv(output_path, index=False)
    print(f"Cleaned dataset saved as CSV to: {output_path}")


if __name__ == "__main__":
    main()
