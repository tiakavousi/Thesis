import os
import sys
import gc
import torch
import pandas as pd
from pathlib import Path
from transformers import AutoModelForSequenceClassification, AutoTokenizer
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

# Label mapping: model outputs 0/1/2 → final labels -1/0/1
LABEL_MAP = {0: -1, 1: 0, 2: 1}

# Model loader
def load_model_and_tokenizer(model_name, model_config):
    print(f"[INFO] 🔄 Loading {model_name} model and tokenizer...")

    model_dir = model_config["model_save_path"]

    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    model.to(DEVICE)
    model.eval()

    assert model.config.num_labels == 3, f"[❌] Model '{model_name}' does not have 3 output labels."

    print(f"[INFO] ✅ {model_name} model loaded and ready with {model.config.num_labels} labels.")
    return model, tokenizer

# Prediction
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

# Main loop
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

                # Map model predictions to -1, 0, 1
                mapped_preds = [LABEL_MAP[p] for p in preds]

                df[f"{model_name}_sentiment_label"] = mapped_preds
                df[f"{model_name}_confidence"] = confs
                print(f"[🔒] Output will be saved to: {output_path}")
                df.to_csv(output_path, index=False)
                print(f"[💾] Sentiment saved to: {output_path}")

                del texts, preds, confs, df
                gc.collect()
                print(f"[🧹] Memory cleared after file: {file_name}")

        del model, tokenizer
        gc.collect()
        print(f"[🧽] Model {model_name} removed from memory.\n")

    print("\n[🎉] All models and repositories processed.")

# Entry point
if __name__ == "__main__":
    run_inference()
