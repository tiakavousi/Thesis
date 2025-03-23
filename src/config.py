import os
import torch

CONFIG = {
    "github": {
        "token": os.getenv("GITHUB_TOKEN"),
        "base_url": "https://api.github.com",
        "raw_dir": "data/raw",
        "processed_dir": "/Users/tayebekavousi/Desktop/github_sa/data/processed",
        "classified_dir": "/Users/tayebekavousi/Desktop/github_sa/data/classified",
        "repositories": [
            "redis/redis", #1
            "apache/kafka", #2
            "elementary/terminal", #3
            "audacity/audacity", #4
            "deluge-torrent/deluge", #5
            "buildaworldnet/IrrlichtBAW", #6
            "linuxmint/cinnamon-desktop", #7
            "linuxmint/cinnamon", #8
            "qBittorrent/qBittorrent", #9
            "CivMC/Civ", #10
            "aquasecurity/trivy" #11
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
    },
    
    # Model training parameters
    "training": {
        "batch_size": 16,
        "max_length": 128,
        "learning_rate": 2e-5,
        "epochs": 3,
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
            "model_save_path": "/Users/tayebekavousi/Desktop/github_sa/saved_models/deberta_best_model_epoch2"
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
    },
    # Add these keys under CONFIG in config.py
    "sentiment_classification": {
        "input_files": {
            "comments": "data/raw/comments_clean.csv",
            "reviews": "data/raw/reviews_clean.csv",
            "review_comments": "data/raw/review_comments_clean.csv",
            "pull_requests": "data/raw/pull_request_clean.csv"
        },
        "output_files": {
            "comments": "data/processed/comments_sentiment.csv",
            "reviews": "data/processed/reviews_sentiment.csv",
            "review_comments": "data/processed/review_comments_sentiment.csv",
            "pull_requests": "data/processed/pull_request_sentiment.csv"
        },
        "text_columns": {
            "comments": "body",
            "reviews": "body",
            "review_comments": "body",
            "pull_requests": "title"
        }
    }

}