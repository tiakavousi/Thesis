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
  - Applies best-performing model to classify comment/review sentiment as Positive or Negative

- **Statistical Analysis:**  
  - Aggregates sentiment data per PR (e.g., negative ratio)
  - Performs binary logistic regression to test effect of sentiment on PR success
  - Includes support for data balancing, standardization, odds ratio interpretation, and visualization (ROC, confusion matrix, etc.)

- **Sanity Checking:**  
  - Manually verifies model predictions on selected comments/reviews
  - Compares model output with human annotations to check accuracy

## Project Structure
```
github_sentiment/
├── data/
│   ├── raw/                     # Original GitHub data dumps
│   ├── processed/               # Cleaned and sentiment-labeled files
│   ├── classified/              # Sentiment classification results per repo
│   ├── features/                # Feature engineered datasets for modeling
│   └── manual_classification/   # Sanity-check files and human-labeled data
├── docs/                        # Documentation and research notes
├── experiments/                
│   ├── data_collection/         # API rate check, CSV counters
│   ├── sentiment_classification/
│   ├── statistical_analysis/    # Feature aggregation and model runner
│   └── visualizations/          # ROC, confusion, and coefficient plots
├── models/                      # Saved transformer models
├── notebooks/                   # Jupyter notebooks for EDA and testing
├── src/
│   ├── config.py                # Configuration (paths, model settings)
│   ├── data_collection/         # GitHub API interaction
│   ├── data_processing/         # Cleaning logic and filters
│   ├── modeling/                # Sentiment training, evaluation
│   └── analysis/                # Logistic regression & metrics
├── requirements.txt
└── README.md
```

## Configuration
The project configuration (e.g., API tokens, file paths, repository names) is managed in `src/config.py`

## Prerequisites
- Python 3.7 or later
- A GitHub personal access token with sufficient permissions to access repository data

---
# Installation

## Create and activate a virtual environment:
```
python -m venv venv
source venv/bin/activate
```

## Install dependencies:
```
pip install -r requirements.txt
```

## Set your GitHub token environment variable:
```
export GITHUB_TOKEN=your_personal_access_token
```

---
# How to Run

## 1. Data Collection
```
python -m src.data_collection.github_collector
```
Saves CSVs in `data/raw/<repo>/`

## 2. Data Cleaning & Bot Filtering
```
python -m src.data_processing.clean_raw_data
```
Saves cleaned files to `data/processed/`

## 3. Sentiment Classification
```
python -m src.modeling.sentiment_classifier --train
python -m src.modeling.sentiment_classifier --evaluate
```
Saves results to `models/` and `data/classified/`

## 4. Sentiment Feature Aggregation
```
python experiments/statistical_analysis/sentiment_aggregator.py
```
Generates per-repo PR-level features (e.g., negativity ratios)

## 5. Merge All Feature Files
```
python experiments/statistical_analysis/merge_features.py
```
Creates `data/features/all_repo_features.csv`

## 6. Logistic Regression Analysis
```
python experiments/statistical_analysis/run_logistic_model.py
```
Runs logistic model using standardized features and optional dataset balancing.

## 7. Visualization & Evaluation
```
python experiments/statistical_analysis/visualize_model.py
```
Generates ROC curve, confusion matrix, coefficient plot, and probability distribution.

## 8. Manual Sentiment Sanity Check (Optional)
To verify model accuracy on real PR comments:
- Prepare a sample CSV: `data/manual_classification/<repo>/sampled_comments.csv`
- Add manual sentiment labels → `labeled_comments.csv`
- Evaluate with: `evaluate_labels.py`

---
# Deactivation
```
deactivate
```

---
# Additional Notes
- **Documentation**: Found in `docs/`
- **Notebooks**: Exploratory and model analysis in `notebooks/`
- **Manual Checks**: Sanity check and quality control in `manual_classification/`

---
## Summary of Collected Data

| Repository                         | Pull Requests | Comments | Reviews |
|------------------------------------|---------------|----------|---------|
| kubernetes/kubernetes              | 100           | 9393     | 12988   |
| redis/redis                        | 100           | 1980     | 6583    |
| apache/kafka                       | 100           | 4293     | 8506    |
| elementary/terminal                | 100           | 487      | 445     |
| audacity/audacity                  | 100           | 2252     | 4764    |
| deluge-torrent/deluge              | 100           | 786      | 411     |
| buildaworldnet/IrrlichtBAW         | 100           | 562      | 5655    |
| linuxmint/cinnamon-desktop         | 100           | 84       | 10      |
| qBittorrent/qBittorrent            | 100           | 5015     | 5809    |
| CivMC/Civ                          | 100           | 157      | 215     |
|------------------------------------|---------------|----------|---------|  
| **Total**                          | **1000**      | **25009**| **45386**|

