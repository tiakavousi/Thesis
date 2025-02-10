# GitHub PR Sentiment Analysis & Data Pipeline

This project is a modular Python-based pipeline that collects, processes, and analyzes GitHub pull request (PR) data. It retrieves the top commented closed PRs (including their comments and reviews) from multiple repositories, cleans and processes the data, applies a BERT-based sentiment classifier on the comments/reviews, and performs statistical analysis on both the raw and processed data.

## Features

- **Data Collection:**  
  - Retrieves the top commented closed PRs from GitHub and corresponding metadata (title, author, creation/closure dates, state, etc.).
  - Gathers all associated comments and reviews until the PR is closed or merged.

- **Data Processing & Cleaning:**  
  - Cleans the raw CSV comments and reviews data

- **Sentiment Analysis:**  
  - Utilizes a BERT-based sentiment classifier to label the tone of comments/reviews.

- **Statistical Analysis:**  
  - Performs statistical analysis on both raw and processed data.

## Project Structure
```
github_sentiment/
├── data/
│   ├── raw/                # Original GitHub data dumps
│   └── processed/          # Final cleaned and processed data
├── docs/                   # Documentation files and other related works
├── experiments/            # Experimental scripts
├── models/                 # Saved or serialized models
├── notebooks/              # Jupyter notebooks for exploration and visualization
├── src/                    # Source code organized into packages
│   ├── __init__.py
│   ├── config.py           # Configuration settings (e.g., API tokens, file paths)
│   ├── data_collection/    # Module for collecting data from GitHub
│   │   ├── __init__.py
│   │   └── github_collector.py
│   ├── data_processing/    # Module for processing and cleaning data
│   │   ├── __init__.py
│   │   └── clean_data.py
│   ├── modeling/           # Module for building and training the BERT sentiment classifier
│   │   ├── __init__.py
│   │   └── sentiment_classifier.py
│   └── analysis/           # Module for statistical analysis and evaluation
│       ├── __init__.py
│       └── statistical_analysis.py
├── requirements.txt        # Python package dependencies
└── README.md               # Project overview and instructions
```

## Configuration
The project configuration (e.g., API tokens, file paths, repository names) is managed in `src/config.py`

## Prerequisites
- Python 3.7 or later
- A GitHub personal access token with sufficient permissions to access repository data
========================================
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
========================================
# How to Run
## Data Collection
```
python -m src.data_collection.github_collector

```
This will save CSV files in the data/raw/<repository>/ subdirectories.

## Data Processing & Cleaning
```
python -m src.data_processing.clean_data
```
The cleaned data will be saved in data/processed/ as configured.

## Sentiment Analysis
Train or evaluate the BERT-based sentiment classifier:
```
python -m src.modeling.sentiment_classifier --train
# or for evaluation
python -m src.modeling.sentiment_classifier --evaluate
```
Trained models are stored in the models/ directory.

## Statistical Analysis
```
python -m src.analysis.statistical_analysis
```
========================================
# Deactivation
When finished, you can deactivate your virtual environment:
```
deactivate
```
========================================
# Additional Notes
- **Documentation**: Additional project documentation, design notes and related papers can be found in the docs/ folder.
- **Notebooks & Experiments**: For exploratory data analysis and additional experiments, check out the notebooks in the notebooks/ and scripts in the experiments/ folders.

========================================
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
|Totall                              | 1000          | 25009    | 45386   |


