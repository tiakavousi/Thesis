import os
import sys
import torch
import pandas as pd
from torch.nn.functional import softmax
from transformers import (
    AutoTokenizer,
    DebertaForSequenceClassification, DebertaTokenizerFast,
    RobertaForSequenceClassification, RobertaTokenizerFast,
    DistilBertForSequenceClassification, DistilBertTokenizerFast
)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.config_three_class import CONFIG_3CLASS

# === CONFIG ===
DEVICE = CONFIG_3CLASS["training"]["device"]
BATCH_SIZE = CONFIG_3CLASS["training"]["batch_size"]
MAX_LEN = CONFIG_3CLASS["training"]["max_length"]
MODELS = CONFIG_3CLASS["models"]
CLASS_NAMES = CONFIG_3CLASS["dataset"]["class_names"]
LABEL_MAP = {i: label for i, label in enumerate(CLASS_NAMES)}

# === Paths ===
INPUT_FILE = "data/merged_for_voting/merged_voting_input.csv"
OUTPUT_FILE = "data/merged_for_voting/model_predictions.csv"

# === Load Models ===
def load_model_and_tokenizer(model_name, model_path):
    checkpoint_path = os.path.join(model_path, "model_best_f1_0.8954.pt")
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

    if model_name == "deberta":
        model = DebertaForSequenceClassification.from_pretrained("microsoft/deberta-base", num_labels=3)
        tokenizer = DebertaTokenizerFast.from_pretrained("microsoft/deberta-base")

    elif model_name == "codebert":
        model = RobertaForSequenceClassification.from_pretrained("microsoft/codebert-base", num_labels=3)
        tokenizer = RobertaTokenizerFast.from_pretrained("microsoft/codebert-base")

    elif model_name == "distilbert":
        model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=3)
        tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    else:
        raise ValueError(f"Unknown model name: {model_name}")

    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.to(DEVICE)
    model.eval()
    return model, tokenizer

# === Load All Models ===
models = {
    name: load_model_and_tokenizer(name, cfg["model_save_path"])
    for name, cfg in MODELS.items()
}

# === Predict Function ===
def predict(model, tokenizer, texts):
    preds, confs = [], []
    for i in range(0, len(texts), BATCH_SIZE):
        batch_texts = texts[i:i+BATCH_SIZE]
        encodings = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt"
        ).to(DEVICE)
        with torch.no_grad():
            outputs = model(**encodings)
            probs = softmax(outputs.logits, dim=1)
            max_probs, pred_labels = torch.max(probs, dim=1)
        preds.extend(pred_labels.cpu().tolist())
        confs.extend(max_probs.cpu().tolist())
    return preds, confs

# === Load Input Data ===
df = pd.read_csv(INPUT_FILE)
texts = df["body"].astype(str).tolist()

# === Run Predictions ===
for model_name, (model, tokenizer) in models.items():
    print(f"[INFO] Predicting with {model_name}...")
    preds, confs = predict(model, tokenizer, texts)
    df[f"{model_name}_label"] = preds
    df[f"{model_name}_confidence"] = confs

# === Save Output ===
df.to_csv(OUTPUT_FILE, index=False)
print(f"[✅] Saved model predictions to: {OUTPUT_FILE}")
