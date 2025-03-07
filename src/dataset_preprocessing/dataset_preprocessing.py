import pandas as pd
import re
import emoji

def clean_text(text, lower_case=False):
    """
    Clean and preprocess the input text for DistilBERT.

    Best practices for DistilBERT include minimal cleaning:
    - Remove URLs and HTML tags.
    - Convert emojis to textual representations so their sentiment can be leveraged.
    - Normalize whitespace.
    - Optionally, insert spaces in some concatenated words 
      (heuristic: if a lowercase letter is directly followed by an uppercase letter).
    - Optionally convert text to lower-case if using a cased model 
      (e.g., 'distilbert-base-uncased').

    Parameters
    ----------
    text : str
        The raw input text to be cleaned.
    lower_case : bool, optional (default=False)
        Whether to convert text to lower-case.

    Returns
    -------
    str
        The cleaned text.
    """
    # Ensure the text is a string and remove leading/trailing whitespace
    text = str(text).strip()
    
    # Remove URLs (patterns starting with http, https, or www)
    text = re.sub(r'http\S+|www\.\S+', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Convert emojis to text (e.g., 😊 -> :smiling_face_with_smiling_eyes:)
    text = emoji.demojize(text)
    
    # Normalize multiple whitespace into one space
    text = re.sub(r'\s+', ' ', text)
    
    # Insert a space before uppercase letters if preceded by a lowercase letter
    text = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', text)
    
    # Optionally convert text to lower case
    if lower_case:
        text = text.lower()
    
    return text

def main():
    # 1) Load the dataset from Excel
    file_path = "/Users/tayebekavousi/Desktop/github_sa/datasets/binary/code-review-dataset-full.xlsx"
    df = pd.read_excel(file_path)
    
    # 2) Remove rows where 'message' is empty or NaN
    df = df.dropna(subset=["message"])                 # Remove rows with NaN in 'message'
    df = df[df["message"].str.strip() != ""]           # Remove rows where 'message' is just whitespace
    
    # 3) Clean the text in the 'message' column
    #    (set lower_case=True if you want to use an uncased DistilBERT model)
    df["message"] = df["message"].apply(lambda x: clean_text(x, lower_case=False))
    
    # Remove rows that might have become empty after cleaning
    df = df[df["message"].str.strip() != ""]
    
    # 4) Find the minimum class count (for downsampling)
    class_counts = df["is_toxic"].value_counts()
    min_count = class_counts.min()
    
    if min_count == 0:
        raise ValueError("At least one class has 0 rows. Cannot downsample to 0.")
    
    # 5) Downsample each class to the min_count
    balanced_df = (
        df.groupby("is_toxic")
          .apply(lambda x: x.sample(n=min_count, random_state=42))
          .reset_index(drop=True)
    )
    
    # 6) Save the balanced dataset to a CSV file
    output_path = "/Users/tayebekavousi/Desktop/github_sa/datasets/preprocessed/code-review-dataset-balanced.csv"
    balanced_df.to_csv(output_path, index=False)
    print(f"Balanced dataset saved to: {output_path}")

if __name__ == "__main__":
    main()
