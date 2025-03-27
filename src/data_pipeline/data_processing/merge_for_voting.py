import os
import pandas as pd
from src.config import CONFIG

# === Config Inputs ===
REPOS = CONFIG["github"]["repositories"]
PROCESSED_DIR = CONFIG["github"]["processed_dir"]
OUTPUT_DIR = os.path.join("data", "merged_for_voting")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Input File Names from Config (cleaned files)
FILE_NAMES = CONFIG["github"]["file_names"]
COMMENT_FILE = FILE_NAMES["comments"].replace(".csv", "_clean.csv")
REVIEW_FILE = FILE_NAMES["reviews"].replace(".csv", "_clean.csv")
REVIEW_COMMENT_FILE = FILE_NAMES["review_comments"].replace(".csv", "_clean.csv")
PR_FILE = FILE_NAMES["pull_requests"].replace(".csv", "_clean.csv")

# === Column Definitions ===
COMMON_COLUMNS = ["id", "pr_number", "body"]
PR_COLUMNS = ["id", "number", "title", "comments", "state"]

# === File Map for Merging (comments, review_comments, reviews)
FILE_MAP = {
    COMMENT_FILE: COMMON_COLUMNS,
    REVIEW_COMMENT_FILE: COMMON_COLUMNS,
    REVIEW_FILE: COMMON_COLUMNS,
}

# === Merge Comment/Review Data ===
merged_dfs = []

print(f"[INFO] Merging sentiment-ready data from: {PROCESSED_DIR}")

for repo in REPOS:
    owner, name = repo.split("/")
    repo_dir = os.path.join(PROCESSED_DIR, owner, name)

    for file_name, columns in FILE_MAP.items():
        path = os.path.join(repo_dir, file_name)
        if not os.path.exists(path):
            print(f"[⚠️] Missing: {path}")
            continue

        try:
            df = pd.read_csv(path, usecols=columns)
            df["repo"] = repo
            merged_dfs.append(df)
            print(f"[✓] Merged {file_name} for {repo}")
        except Exception as e:
            print(f"[ERROR] Could not read {path}: {e}")

# === Save Merged Voting Input ===
if merged_dfs:
    merged_data = pd.concat(merged_dfs, ignore_index=True)
    output_path = os.path.join(OUTPUT_DIR, "merged_voting_input.csv")
    merged_data.to_csv(output_path, index=False)
    print(f"[✅] Saved merged voting input to: {output_path}")
else:
    print("[⚠️] No data merged. Check your input files.")

# === Process Pull Requests Separately ===
pr_dfs = []

for repo in REPOS:
    owner, name = repo.split("/")
    repo_dir = os.path.join(PROCESSED_DIR, owner, name)
    path = os.path.join(repo_dir, PR_FILE)

    if not os.path.exists(path):
        print(f"[⚠️] Missing PR file: {path}")
        continue

    try:
        df = pd.read_csv(path, usecols=PR_COLUMNS)
        df["repo"] = repo
        pr_dfs.append(df)
        print(f"[✓] Merged PR data for {repo}")
    except Exception as e:
        print(f"[ERROR] Could not read PR file {path}: {e}")

if pr_dfs:
    pr_merged = pd.concat(pr_dfs, ignore_index=True)
    output_path = os.path.join(OUTPUT_DIR, "merged_pull_requests.csv")  # 👈 renamed here
    pr_merged.to_csv(output_path, index=False)
    print(f"[✅] Saved merged PR sentiment to: {output_path}")
else:
    print("[⚠️] No PR sentiment data merged.")
