import os
import sys
import torch
import pandas as pd
from torch.nn.functional import softmax
from transformers import (
    DebertaForSequenceClassification, DebertaTokenizerFast,
    RobertaForSequenceClassification, RobertaTokenizerFast,
    DistilBertForSequenceClassification, DistilBertTokenizerFast
)

# Add project root to path and import config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.config_three_class import CONFIG_3CLASS

# === CONFIG ===
DEVICE = CONFIG_3CLASS["training"]["device"]
BATCH_SIZE = CONFIG_3CLASS["training"]["batch_size"]
MAX_LEN = CONFIG_3CLASS["training"]["max_length"]
MODELS = CONFIG_3CLASS["models"]
CLASS_NAMES = CONFIG_3CLASS["dataset"]["class_names"]
INFERENCE = CONFIG_3CLASS["inference"]
INPUT_FILE = INFERENCE["input_file"]
OUTPUT_FILE = INFERENCE["output_file"]
TEXT_COLUMN = INFERENCE["text_column"]

print(f"\n📦 Device selected: {DEVICE}")
print(f"📚 Loading models: {list(MODELS.keys())}")
print(f"📄 Input file: {INPUT_FILE}")
print(f"📝 Text column: '{TEXT_COLUMN}'\n")

# === Load Models ===
def load_model_and_tokenizer(model_name, model_config):
    checkpoint_path = model_config["model_save_path"]
    pretrained_name = model_config["pretrained_model_name"]

    print(f"🔧 Loading {model_name.upper()} from checkpoint:\n   {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

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

    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.to(DEVICE)
    model.eval()
    print(f"✅ {model_name} model and tokenizer loaded.\n")
    return model, tokenizer

# === Load All Models ===
print("🚀 Loading all models...")
models = {
    name: load_model_and_tokenizer(name, cfg)
    for name, cfg in MODELS.items()
}
print("✅ All models loaded successfully.\n")

# === Predict Function ===
def predict(model, tokenizer, texts):
    preds, confs = [], []
    total = len(texts)
    for i in range(0, total, BATCH_SIZE):
        batch_texts = texts[i:i + BATCH_SIZE]
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

        print(f"🧠 Processed batch {i // BATCH_SIZE + 1} of {((total - 1) // BATCH_SIZE + 1)}")

    return preds, confs

# === Load Input Data ===
print("📥 Loading input data...")
df = pd.read_csv(INPUT_FILE)
texts = df[TEXT_COLUMN].astype(str).tolist()
print(f"✅ Loaded {len(texts)} texts for prediction.\n")

# === Run Predictions ===
for model_name, (model, tokenizer) in models.items():
    print(f"🔮 Running predictions with {model_name.upper()}...")
    preds, confs = predict(model, tokenizer, texts)
    df[f"{model_name}_label"] = preds
    df[f"{model_name}_confidence"] = confs
    print(f"✅ Finished predictions for {model_name.upper()}.\n")

# === Save Output ===
print(f"💾 Saving predictions to: {OUTPUT_FILE} ...")
df.to_csv(OUTPUT_FILE, index=False)
print(f"✅ Saved model predictions to: {OUTPUT_FILE}")
print("🎉 All done!")
