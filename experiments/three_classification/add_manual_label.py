import os
import pandas as pd
import json
from src.config import CONFIG

# Paths
MANUAL_COMMENTS_PATH = "data/reviewed/comments_manual_reviewed.csv"
MANUAL_PRS_PATH = "data/reviewed/pull_requests_manual_reviewd.csv"
MERGED_DIR = "data/merged_voting"
OUTPUT_DIR = "data/labeled_with_manual"

# Load manual labels
comments_manual = pd.read_csv(MANUAL_COMMENTS_PATH)
prs_manual = pd.read_csv(MANUAL_PRS_PATH)
manual_all = pd.concat([comments_manual, prs_manual], ignore_index=True)
manual_all = manual_all.rename(columns={"manual_sntiment_label": "manual_label"})
manual_all = manual_all.set_index(["repository", "file_type", "id"])

# Get repositories from config
repositories = CONFIG["github"]["repositories"]

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# File types and corresponding file names
file_map = {
    "comments": "comments_classified.csv",
    "review_comments": "review_comments_classified.csv",
    "reviews": "reviews_classified.csv",
    "pull_requests": "pull_requests_classified.csv",
}

# Process each repo and file type
for repo in repositories:
    owner, repo_name = repo.split("/")
    for file_type, filename in file_map.items():
        input_path = os.path.join(MERGED_DIR, owner, repo_name, filename)
        if not os.path.exists(input_path):
            print(f"❌ Missing file: {input_path}")
            continue

        df = pd.read_csv(input_path)

        # Determine key columns based on file type
        key_cols = ["repository", "file_type", "id"]
        df["repository"] = repo
        df["file_type"] = file_type

        # Join manual labels
        df.set_index(key_cols, inplace=True)
        df["manual_label"] = manual_all["manual_label"] if manual_all.index.isin(df.index).any() else 2
        df["manual_label"] = manual_all["manual_label"].reindex(df.index).fillna(2).astype(int)
        df.reset_index(inplace=True)

        # Prepare output path
        output_repo_dir = os.path.join(OUTPUT_DIR, owner, repo_name)
        os.makedirs(output_repo_dir, exist_ok=True)
        output_path = os.path.join(output_repo_dir, filename)

        # Save
        df.to_csv(output_path, index=False)
        print(f"✅ Saved with manual labels: {output_path}")