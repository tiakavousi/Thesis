import os
import torch

CONFIG = {
    "github": {
        "token": os.getenv("GITHUB_TOKEN"),
        "base_url": "https://api.github.com",
        "raw_dir": "data/raw",
        "repositories": [
            "kubernetes/kubernetes", #1
            "redis/redis", #2
            "apache/kafka", #3
            "elementary/terminal", #4
            "audacity/audacity", #5
            "deluge-torrent/deluge", #6
            "buildaworldnet/IrrlichtBAW", #7
            "linuxmint/cinnamon-desktop", #8
            "linuxmint/cinnamon", #8
            "qBittorrent/qBittorrent", #9
            "CivMC/Civ", #10
        ],
        "pull_requests": {
            "top_n": 100  # Number of PRs to retrieve
        },
        "file_names": {
            "pull_requests": "pull_requests.csv",
            "comments": "comments.csv",
            "reviews": "reviews.csv",
            "review_comments": "review_comments.csv",
            "summary_data": "summary.csv"
        },
       "processed_dir":"data/processed"
    },
    
    # Common model training parameters
    "training": {
        # "batch_size": 16, #VERSION 1
        # "max_length": 128, #VERSION 1
        # "learning_rate": 2e-5, #VERSION 1
        # "epochs": 3,
        "batch_size": 32,  #VERSION 2
        "max_length": 256, #VERSION 2
        "learning_rate": 1e-5, #VERSION 2
        "lr_scheduler": {
            "type": "linear",
            "warmup_ratio": 0.1
        },
        "epochs": 5, #VERSION 2
        "gradient_accumulation_steps": 2, #VERSION 2
        "weight_decay": 0.01,  # L2 regularization #VERSION 2
        "dropout": 0.1,  # For additional regularization    #VERSION 2
        "early_stopping_patience": 2, 
        "early_stopping_metric": "val_f1",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    },
    
    # Model-specific configurations
    "models": {
        "distilbert": {
            "pretrained_model_name": "distilbert-base-uncased",
            "model_save_path": "./saved_models/distilbert_finetuned"
        },
        "codebert": {
            "pretrained_model_name": "microsoft/codebert-base",
            "model_save_path": "./saved_models/codebert_finetuned"
        },
        "deberta": {
            "pretrained_model_name": "microsoft/deberta-base",
            "model_save_path": "./saved_models/deberta_finetuned",
            "layer_freezing": 2
        }
    },
    
    # Dataset configuration
    "dataset": {
        "train_ratio": 0.75,
        "val_ratio": 0.15,
        "test_ratio": 0.10,
        "random_seed": 42,
        "raw_combined_dataset": "datasets/combined_dataset.xlsx",
        "raw_toxiCR_dataset": "datasets/code-review-dataset-full.xlsx",
        "dataset_path": "datasets/preprocessed/combined_dataset-balanced.csv",
        "class_names": ["Negative", "Positive"],
        "num_labels": 2,
    }
}