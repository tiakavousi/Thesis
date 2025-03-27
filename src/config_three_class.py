import torch
from pathlib import Path

# Dynamically resolve the project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # github_sa/

CONFIG_3CLASS = {
    "training": {
        "batch_size": 32,
        "max_length": 128,
        "learning_rate": 1e-5,
        "epochs": 5,
        "early_stopping_patience": 2,
        "early_stopping_metric": "val_f1",
        "device": "mps" if torch.backends.mps.is_available() else "cpu"
        # "device": "cuda" if torch.cuda.is_available() else "cpu",
    },
    "models": {
        "deberta": {
            "pretrained_model_name": "microsoft/deberta-base",
            "model_save_path": str(PROJECT_ROOT / "saved_models" / "deberta_3class_v02")
        },
        "codebert": {
            "pretrained_model_name": "microsoft/codebert-base",
            "model_save_path": str(PROJECT_ROOT / "saved_models" / "codebert_3class")
        },
        "distilbert": {
            "pretrained_model_name": "distilbert-base-uncased",
            "model_save_path": str(PROJECT_ROOT / "saved_models" / "distilbert_3class")
        },
    },
    "dataset": {
        "train_ratio": 0.75,
        "val_ratio": 0.15,
        "test_ratio": 0.10,
        "random_seed": 42,
        "raw_combined_dataset": str(PROJECT_ROOT / "datasets" / "raw" / "combined_DeepSentimentSECrossPlatform.csv"),
        "dataset_path": str(PROJECT_ROOT / "datasets" / "preprocessed" / "combined_DeepSentimentSECrossPlatform.csv"),
        "class_names": ["Negative", "Neutral", "Positive"],
        "num_labels": 3,
    },
    "evaluation": {
        "report_save_path": str(PROJECT_ROOT / "saved_models" / "codebert_3class" / "evaluation_result_codebert")
    }
}
