# GitHub PR Sentiment Analysis & Data Pipeline

This project is a modular Python-based pipeline that collects, processes, and analyzes GitHub pull request (PR) data. It retrieves the top commented closed PRs (including their comments and reviews) from multiple repositories, cleans and processes the data, applies a BERT-based sentiment classifier on the comments/reviews, and performs statistical analysis on both the raw and processed data.

## Features

- **Data Collection:**  
  - Retrieves the top commented closed PRs from GitHub and corresponding metadata (title, author, creation/closure dates, state, etc.).
  - Gathers all associated comments and reviews until the PR is closed or merged.

- **Data Processing & Cleaning:**  
  - Cleans the raw CSV comments and reviews data
  - Filters bot-generated messages, commands, and irrelevant noise

- **Sentiment Analysis:**  
  - Fine-tunes multiple transformer models (DistilBERT, CodeBERT, DeBERTa)
  - Uses majority voting among models
  - Classifyes the sentiment into : Positive, Neutral and Negative

- **Statistical Analysis:**
  - Aggregates sentiment data per PR (e.g., negativity ratio)
  - Performs binary logistic regression to test effect of sentiment on PR success

# Dataset Output sentiGit.csv:
The resulting merged sentiment dataset from this pipeline is saved as sentiGit.csv.
It consolidates 72355 labeled text from PR comments, reviews, inline review comments, and PR titles across multiple repositories.
Each entry includes a unique ID, text, and the sentiment label.
Sentiment labels are derived as a result of fine tuning three transformer models (DeBERTa, DistilBERT and CodeBERT) and Majority Voting ensemble model.

## Project Structure
```
github_sa/
├── data/                          # Raw, processed, and classified data
│   ├── raw/                       # Unfiltered GitHub data
│   ├── processed/                 # Cleaned text ready for sentiment analysis
│   ├── classified/                # Model-predicted sentiment results
│   ├── labeled_with_manual/       # Final labeled data (includes manual review)
│   ├── merged_voting/             # Per-text outputs from all 3 models
│   ├── pr_sentiment_aggregated/   # PR-level aggregated sentiment features
│   └── reviewed/                  # Sanity check and manually validated outputs

├── datasets/                      # External datasets for training and validation
│   ├── raw/                       # Original datasets like DeepSentimentSE
│   └── preprocessed/              # Cleaned and class-balanced datasets

├── experiments/
│   ├── data_collection/           # Scripts for GitHub data scraping & stats
│   └── three_classification/      # 3-class sentiment classification pipeline

├── notebooks/                     # Jupyter notebooks for analysis & testing
│   ├── dataset_exploration/       # Exploratory analysis of datasets
│   ├── fine_tuning/               # Model training and evaluation
│   └── load_and_prediction/       # Load saved models and run inference

├── saved_models/                  # Fine-tuned transformer models

├── src/                           # Source code
│   ├── config.py                  # Main configuration
│   ├── config_three_class.py      # Config for 3-class models
│   ├── data_pipeline/             # Data loading, cleaning, preprocessing
│   ├── models/                    # Model wrappers and runners
│   └── training/                  # Training and evaluation utilities

├── sentiGit.csv                   # Final labeled dataset for public use
├── requirements.txt               # Python dependencies
├── readme.md                      # Project overview and instructions
└── report_ganarator/              # Scripts for generating model reports

```

## Configuration
The project configuration (e.g., API tokens, file paths, repository names) is managed in `src/config.py` and `src/config_three_class.py`

## Create and activate a virtual environment:
```
python -m venv venv
source venv/bin/activate
```

## Install dependencies:
```
pip install -r requirements.txt
```

# Deactivation
```
deactivate
```

