import os
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.config import CONFIG


def load_model_from_config():
    """
    Load the model and tokenizer from the absolute path specified in the config.
    The model directory must contain:
      - config.json
      - pytorch_model.bin (your model weights)
      - a 'tokenizer' subdirectory with all required tokenizer files
    """
    model_path = CONFIG["models"]["deberta"]["model_save_path"]
    print(f"[INFO] Loading model from: {model_path}")

    model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
    
    # Load tokenizer explicitly from the 'tokenizer' subdirectory
    tokenizer_dir = os.path.join(model_path, "tokenizer")
    if not os.path.exists(tokenizer_dir):
        raise FileNotFoundError(f"Tokenizer directory not found at {tokenizer_dir}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
    
    model.eval()
    print("[INFO] Model loaded successfully and set to evaluation mode.")
    return model, tokenizer


class SentimentPredictor:
    def __init__(self):
        """
        Initialize the predictor by loading the model and tokenizer using the absolute path from config.
        """
        print("[INFO] Initializing SentimentPredictor...")
        self.model, self.tokenizer = load_model_from_config()
        self.device = torch.device(CONFIG["training"]["device"])
        self.model.to(self.device)
        self.batch_size = CONFIG["training"]["batch_size"]
        self.max_length = CONFIG["training"]["max_length"]
        print("[INFO] SentimentPredictor initialized.\n")
    
    def predict(self, texts, batch_size=None):
        """
        Predict sentiment labels for a list of texts using batch processing.
        """
        if batch_size is None:
            batch_size = self.batch_size
        
        predictions = []
        total_batches = (len(texts) + batch_size - 1) // batch_size
        print(f"[INFO] Predicting sentiment for {len(texts)} texts in {total_batches} batch(es) (batch size: {batch_size})...")
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            encodings = self.tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            encodings = {key: val.to(self.device) for key, val in encodings.items()}
            with torch.no_grad():
                outputs = self.model(**encodings)
                logits = outputs.logits
                batch_preds = torch.argmax(logits, dim=1)
            predictions.extend(batch_preds.cpu().numpy().tolist())
            print(f"[INFO] Processed batch {(i // batch_size) + 1} of {total_batches}.")
        
        print("[INFO] Prediction complete.\n")
        return predictions
    
    def process_file(self, input_file, text_column, output_file, batch_size=None):
        """
        Process a CSV file to predict sentiment and save the results to a new file.
        """
        print(f"[INFO] Starting processing of file: {input_file}")
        try:
            df = pd.read_csv(input_file)
            print(f"[INFO] Loaded {len(df)} record(s) from '{input_file}'.")
        except FileNotFoundError:
            print(f"[WARNING] File not found: '{input_file}'. Skipping this file.\n")
            return
        
        if text_column not in df.columns:
            print(f"[WARNING] Column '{text_column}' not found in '{input_file}'. Skipping this file.\n")
            return
        
        texts = df[text_column].astype(str).tolist()
        preds = self.predict(texts, batch_size=batch_size)
        
        # Map numeric predictions to class names (assumes 0 -> Negative, 1 -> Positive)
        class_names = CONFIG["dataset"]["class_names"]
        df["sentiment"] = [class_names[p] for p in preds]
        
        df.to_csv(output_file, index=False)
        print(f"[INFO] File processed and saved to: '{output_file}'\n")


def main():
    print("[INFO] Starting sentiment classification process...\n")
    predictor = SentimentPredictor()
    
    processed_dir = CONFIG["github"]["processed_dir"]
    classified_dir = CONFIG["github"].get("classified_dir", "data/classified")
    repo_list = CONFIG["github"]["repositories"]
    
    # Define the files to process along with the text column to classify.
    files_to_process = {
        "comments_clean.csv": "body",
        "reviews_clean.csv": "body",
        "review_comments_clean.csv": "body",
        "pull_requests_clean.csv": "title"
    }
    
    for repo in repo_list:
        try:
            owner, repo_name = repo.split("/")
        except ValueError:
            print(f"[WARNING] Repository format error: '{repo}'. Expected format 'owner/repo'. Skipping this repository.\n")
            continue
        
        repo_dir = os.path.join(processed_dir, owner, repo_name)
        if not os.path.exists(repo_dir):
            print(f"[WARNING] Directory not found for repository '{repo}': '{repo_dir}'. Skipping this repository.\n")
            continue
        
        print(f"[INFO] Processing repository: '{repo}' in '{repo_dir}'")
        output_repo_dir = os.path.join(classified_dir, owner, repo_name)
        os.makedirs(output_repo_dir, exist_ok=True)
        print(f"[INFO] Output directory created: '{output_repo_dir}'\n")
        
        for file_name, text_column in files_to_process.items():
            input_file = os.path.join(repo_dir, file_name)
            if file_name.endswith("_clean.csv"):
                output_file = os.path.join(output_repo_dir, file_name.replace("_clean.csv", "_sentiment.csv"))
            else:
                output_file = os.path.join(output_repo_dir, file_name + "_sentiment.csv")
            predictor.process_file(input_file, text_column, output_file)
    
    print("[INFO] Sentiment classification process completed.")


if __name__ == "__main__":
    main()
