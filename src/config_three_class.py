import torch
from pathlib import Path

# Dynamically resolve the project root directory (e.g., /path/to/github_sa)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Main configuration dictionary for 3-class sentiment classification
CONFIG_3CLASS = {
    "training": {
        "batch_size": 32,                  # Number of samples per batch during training
        "max_length": 128,                 # Maximum token length for input sequences
        "learning_rate": 1e-5,             # Learning rate for optimizer
        "epochs": 5,                       # Maximum number of training epochs
        "early_stopping_patience": 2,      # Number of epochs to wait for improvement before stopping
        "early_stopping_metric": "val_f1", # Metric used to monitor early stopping
        "device": "mps" if torch.backends.mps.is_available() else "cpu"  # Use Apple M1 GPU if available, otherwise fallback to CPU
    },

    "models": {
        # Configuration for DeBERTa
        "deberta": {
            "pretrained_model_name": "microsoft/deberta-base",  # HuggingFace model ID
            "model_save_path": str(PROJECT_ROOT / "saved_models" / "deberta_3class"/ "saved_full_model")  # Path to save/load fine-tuned model
        },
        # Configuration for CodeBERT
        "codebert": {
            "pretrained_model_name": "microsoft/codebert-base",
            "model_save_path": str(PROJECT_ROOT / "saved_models" / "codebert_3class" / "saved_full_model")
        },
        # Configuration for DistilBERT
        "distilbert": {
            "pretrained_model_name": "distilbert-base-uncased",
            "model_save_path": str(PROJECT_ROOT / "saved_models" / "distilbert_3class" / "saved_full_model")
        },
    },

    "dataset": {
        "train_ratio": 0.75,  # 75% for training
        "val_ratio": 0.15,    # 15% for validation
        "test_ratio": 0.10,   # 10% for testing
        "random_seed": 42,    # Random seed for reproducibility

        # Path to the raw input dataset (DeepSentimentSECrossPlatform)
        "raw_combined_dataset": str(PROJECT_ROOT / "datasets" / "raw" / "combined_DeepSentimentSECrossPlatform.csv"),

        # Path to the cleaned and preprocessed dataset
        "dataset_path": str(PROJECT_ROOT / "datasets" / "preprocessed" / "combined_DeepSentimentSECrossPlatform.csv"),

        "class_names": ["Negative", "Neutral", "Positive"],  # Label classes for 3-class classification
        "num_labels": 3,  # Total number of sentiment classes
    },

    "evaluation": {
        # Path to save model evaluation report (PDF/HTML)
        "report_save_path": str(PROJECT_ROOT / "saved_models" / "codebert_3class" / "evaluation_result_codebert")
    },

    "inference": {
        # Path to input file for running predictions
        "input_file": str(PROJECT_ROOT / "data" / "merged_for_voting" / "merged_voting_input.csv"),

        # Path to save predicted output with sentiment labels and confidence
        "output_file": str(PROJECT_ROOT / "data" / "merged_for_voting" / "model_predictions.csv"),

        # Column name in the input CSV to be used as input text
        "text_column": "body"
    }
}
