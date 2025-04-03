import os
import sys
import gc
import torch
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer,AutoConfig
from torch.nn.functional import softmax

# Add project root to import config modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.config import CONFIG
from src.config_three_class import CONFIG_3CLASS

# Load settings
DEVICE = torch.device(CONFIG_3CLASS["training"]["device"])
BATCH_SIZE = CONFIG_3CLASS["training"]["batch_size"]
MAX_LEN = CONFIG_3CLASS["training"]["max_length"]
MODELS = CONFIG_3CLASS["models"]

PROCESSED_DIR = CONFIG["github"]["processed_dir"]
CLASSIFIED_DIR = CONFIG["github"]["classified_dir"]
REPOS = CONFIG["github"]["repositories"]

# File mappings: which column to analyze for each file
FILES_TO_PROCESS = {
    "comments_clean.csv": "body",
    "review_comments_clean.csv": "body",
    "reviews_clean.csv": "body",
    "pull_requests_clean.csv": "title"
}

# Map model output to final sentiment labels
LABEL_MAP = {0: -1, 1: 0, 2: 1}

def load_model_and_tokenizer(model_name, model_config):
    print(f"Loading {model_name} best model and tokenizer...")

    model_dir = model_config["model_save_path"]
    best_ckpt_path = os.path.join(model_dir, f"model_best_f1_*.pt")

    # Search for checkpoint files matching the naming pattern
    import glob
    ckpt_files = glob.glob(best_ckpt_path)
    if not ckpt_files:
        raise FileNotFoundError(f"No best model checkpoint found in {model_dir}")
    ckpt_path = ckpt_files[0]  # Use the first matching checkpoint

    # Load model config and tokenizer from saved directory
    config = AutoConfig.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    # Load the model from the selected checkpoint state
    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path=None,
        config=config,
        state_dict=torch.load(ckpt_path, map_location=DEVICE)
    )

    model.to(DEVICE)
    model.eval()  # Set model to inference mode

    # Ensure model is configured for 3-class classification
    assert model.config.num_labels == 3, f"Model '{model_name}' does not have 3 output labels."

    print(f"Best checkpoint for {model_name} loaded from: {ckpt_path}")
    return model, tokenizer


# Prediction
def predict(model, tokenizer, texts):
    print(f"[INFO] Starting prediction on {len(texts)} text entries...")
    all_preds = []  # Store predicted class indices
    all_confs = []  # Store confidence scores for predictions

    # Process texts in batches
    for i in range(0, len(texts), BATCH_SIZE):
        print(f"  → Processing batch {i // BATCH_SIZE + 1}...")
        batch = texts[i:i + BATCH_SIZE]

        # Tokenize the batch of texts
        encodings = tokenizer(
            batch,
            truncation=True,
            padding=True,
            max_length=MAX_LEN,
            return_tensors="pt"
        ).to(DEVICE)

        # Disable gradient tracking for inference
        with torch.no_grad():
            outputs = model(**encodings)
            # Convert logits to probabilities
            probs = softmax(outputs.logits, dim=1) 
            # Get predicted class and its confidence 
            conf, pred = torch.max(probs, dim=1)    
            # Save predictions and confidences to CPU
            all_preds.extend(pred.cpu().tolist())
            all_confs.extend(conf.cpu().tolist())

    print(f"[INFO] ✅ Prediction complete.\n")
    return all_preds, all_confs


# Main loop
def run_inference():
    print("Starting sentiment inference...\n")

    # Iterate over each model defined in the configuration
    for model_name, model_config in MODELS.items():
        print(f"\nUsing model: {model_name}")
        model, tokenizer = load_model_and_tokenizer(model_name, model_config)

        # Iterate over each GitHub repository
        for repo in REPOS:
            print(f"\nProcessing repository: {repo}")
            owner, repo_name = repo.split("/")
            input_repo_path = os.path.join(PROCESSED_DIR, owner, repo_name)
            output_repo_path = os.path.join(CLASSIFIED_DIR, model_name, owner, repo_name)
            os.makedirs(output_repo_path, exist_ok=True)

            # Process each relevant CSV file within the repository
            for file_name, text_column in FILES_TO_PROCESS.items():
                input_path = os.path.join(input_repo_path, file_name)
                output_path = os.path.join(output_repo_path, file_name.replace("_clean.csv", "_sentiment.csv"))

                if not os.path.exists(input_path):
                    print(f"Skipping: File not found: {input_path}")
                    continue

                print(f"[📄] Reading file: {input_path}")
                df = pd.read_csv(input_path)
                if text_column not in df.columns:
                    print(f"Skipping: Column '{text_column}' missing in {input_path}")
                    continue

                print(f"Predicting sentiments for {len(df)} rows...")
                texts = df[text_column].astype(str).tolist()
                preds, confs = predict(model, tokenizer, texts)

                # Convert model predictions to -1, 0, or 1 labels
                mapped_preds = [LABEL_MAP[p] for p in preds]

                # Store predictions and confidence scores in new columns
                df[f"{model_name}_sentiment_label"] = mapped_preds
                df[f"{model_name}_confidence"] = confs

                print(f"Output will be saved to: {output_path}")
                df.to_csv(output_path, index=False)
                print(f"Sentiment saved to: {output_path}")

                # Clean up memory
                del texts, preds, confs, df
                gc.collect()
                print(f"Memory cleared after file: {file_name}")

        # Remove model and tokenizer from memory before moving to the next
        del model, tokenizer
        gc.collect()
        print(f"Model {model_name} removed from memory.\n")

    print("\nAll models and repositories processed.")


# Entry point
if __name__ == "__main__":
    run_inference()
