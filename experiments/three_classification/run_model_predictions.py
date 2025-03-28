import os
import sys
import gc
import torch
import pandas as pd
from pathlib import Path
from transformers import (
    DebertaTokenizerFast, DebertaForSequenceClassification,
    RobertaTokenizerFast, RobertaForSequenceClassification,
    DistilBertTokenizerFast, DistilBertForSequenceClassification
)
from torch.nn.functional import softmax

# Add paths to config files
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.config import CONFIG
from src.config_three_class import CONFIG_3CLASS

# Load config values
DEVICE = torch.device(CONFIG_3CLASS["training"]["device"])
BATCH_SIZE = CONFIG_3CLASS["training"]["batch_size"]
MAX_LEN = CONFIG_3CLASS["training"]["max_length"]
MODELS = CONFIG_3CLASS["models"]

PROCESSED_DIR = CONFIG["github"]["processed_dir"]
CLASSIFIED_DIR = CONFIG["github"]["classified_dir"]
REPOS = CONFIG["github"]["repositories"]

# File and column mappings
FILES_TO_PROCESS = {
    "comments_clean.csv": "body",
    "review_comments_clean.csv": "body",
    "reviews_clean.csv": "body",
    "pull_requests_clean.csv": "title"
}


# Model-specific loader
def load_model_and_tokenizer(model_name, model_config):
    print(f"[INFO] 🔄 Loading {model_name} model and tokenizer...")
    pretrained_name = model_config["pretrained_model_name"]
    checkpoint_path = model_config["model_save_path"]

    if model_name == "deberta":
        model = DebertaForSequenceClassification.from_pretrained(pretrained_name, num_labels=3)
        tokenizer = DebertaTokenizerFast.from_pretrained(pretrained_name)
    elif model_name == "codebert":
        model = RobertaForSequenceClassification.from_pretrained(pretrained_name, num_labels=3)
        tokenizer = RobertaTokenizerFast.from_pretrained(pretrained_name)
    elif model_name == "distilbert":
        model = DistilBertForSequenceClassification.from_pretrained(pretrained_name, num_labels=3)
        tokenizer = DistilBertTokenizerFast.from_pretrained(pretrained_name)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.to(DEVICE)
    model.eval()

    print(f"[INFO] ✅ {model_name} model loaded and ready.")
    return model, tokenizer

# Prediction function
def predict(model, tokenizer, texts):
    print(f"[INFO] Starting prediction on {len(texts)} text entries...")
    all_preds = []
    all_confs = []
    for i in range(0, len(texts), BATCH_SIZE):
        print(f"  → Processing batch {i // BATCH_SIZE + 1}...")
        batch = texts[i:i+BATCH_SIZE]
        encodings = tokenizer(
            batch,
            truncation=True,
            padding=True,
            max_length=MAX_LEN,
            return_tensors="pt"
        ).to(DEVICE)
        with torch.no_grad():
            outputs = model(**encodings)
            probs = softmax(outputs.logits, dim=1)
            conf, pred = torch.max(probs, dim=1)
            all_preds.extend(pred.cpu().tolist())
            all_confs.extend(conf.cpu().tolist())
    print(f"[INFO] ✅ Prediction complete.\n")
    return all_preds, all_confs

# Main inference loop
def run_inference():
    print("[🚀] Starting sentiment inference...\n")
    for model_name, model_config in MODELS.items():
        print(f"\n[🔧] Using model: {model_name}")
        model, tokenizer = load_model_and_tokenizer(model_name, model_config)

        for repo in REPOS:
            print(f"\n[📁] Processing repository: {repo}")
            owner, repo_name = repo.split("/")
            input_repo_path = os.path.join(PROCESSED_DIR, owner, repo_name)
            output_repo_path = os.path.join(CLASSIFIED_DIR, model_name, owner, repo_name)
            os.makedirs(output_repo_path, exist_ok=True)

            for file_name, text_column in FILES_TO_PROCESS.items():
                input_path = os.path.join(input_repo_path, file_name)
                output_path = os.path.join(output_repo_path, file_name.replace("_clean.csv", "_sentiment.csv"))

                if not os.path.exists(input_path):
                    print(f"[⚠️] Skipping: File not found: {input_path}")
                    continue

                print(f"[📄] Reading file: {input_path}")
                df = pd.read_csv(input_path)
                if text_column not in df.columns:
                    print(f"[⚠️] Skipping: Column '{text_column}' missing in {input_path}")
                    continue

                print(f"[✏️] Predicting sentiments for {len(df)} rows...")
                texts = df[text_column].astype(str).tolist()
                preds, confs = predict(model, tokenizer, texts)
                df[f"{model_name}_label"] = preds
                df[f"{model_name}_confidence"] = confs
                df.to_csv(output_path, index=False)
                print(f"[💾] Sentiment saved to: {output_path}")

                # Clear memory after each file
                del texts, preds, confs, df
                gc.collect()
                print(f"[🧹] Memory cleared after file: {file_name}")

        # Clear model & tokenizer from memory after each model
        del model, tokenizer
        gc.collect()
        print(f"[🧽] Model {model_name} removed from memory.\n")

    print("\n[🎉] All models and repositories processed.")

# Execute
if __name__ == "__main__":
    run_inference()
