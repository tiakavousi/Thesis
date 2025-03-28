import os
import pandas as pd
from collections import Counter
import warnings
from src.config import CONFIG

# Load model names from config
model_names = list(CONFIG["models"].keys())

# Input and output base directories
BASE_CLASSIFIED_DIR = "data/classified"
MERGED_OUTPUT_DIR = "data/merged_voting"

# File types and corresponding output base names
file_types = [
    "comments_clean.csv",
    "review_comments_clean.csv",
    "reviews_clean.csv",
    "pull_requests_clean.csv"
]

# Map file base names to their text columns and required metadata columns
file_column_config = {
    "comments":        {"text_col": "body",  "keep_cols": ["id", "pr_number", "body"]},
    "review_comments": {"text_col": "body",  "keep_cols": ["id", "pr_number", "body"]},
    "reviews":         {"text_col": "body",  "keep_cols": ["id", "pr_number", "body"]},
    "pull_requests":   {"text_col": "title", "keep_cols": ["id", "number", "title", "state", "merged_at"]},
}

# Get repositories from config
repositories = CONFIG["github"]["repositories"]

# Majority voting function
def compute_majority_vote(row):
    preds = [row[f"{model}_sentiment_label"] for model in model_names]
    confs = [row[f"{model}_confidence"] for model in model_names]

    if preds[0] == preds[1] == preds[2]:
        majority_label = preds[0]
        decision = "accept"
        reason = "unanimous_agreement"

    else:
        count = Counter(preds)
        most_common, freq = count.most_common(1)[0]

        if freq == 2:
            majority_label = most_common
            decision = "accept"
            reason = "majority_agreement"
        else:
            majority_label = None
            decision = "review"
            reason = "no_majority_disagreement"  # All three disagree

    return pd.Series({
        "majority_label": majority_label,
        "final_decision": decision,
        "decision_reason": reason
    })



print("\n🔄 Starting merging and majority voting process...")

for repo in repositories:
    try:
        owner, repo_name = repo.split("/")
    except ValueError:
        warnings.warn(f"Repository format error: '{repo}'. Expected format 'owner/repo'. Skipping.")
        continue

    print(f"\n📁 Processing repository: {repo}")

    for file_type in file_types:
        base_name = file_type.replace("_clean.csv", "")
        config = file_column_config[base_name]
        text_col = config["text_col"]
        keep_cols = config["keep_cols"]

        print(f"  🔍 Processing file type: {file_type}")
        dfs = {}
        missing_model = False

        for model in model_names:
            file_path = os.path.join(
                BASE_CLASSIFIED_DIR,
                model,
                owner,
                repo_name,
                f"{base_name}_sentiment.csv"
            )
            if not os.path.exists(file_path):
                warnings.warn(f"    ⚠️ File missing for model '{model}': {file_path}. Skipping {repo} - {file_type}.")
                missing_model = True
                break
            print(f"    ✅ Found file: {file_path}")
            dfs[model] = pd.read_csv(file_path)

        if missing_model:
            continue

        nrows = [df.shape[0] for df in dfs.values()]
        if len(set(nrows)) != 1:
            warnings.warn(f"    ⚠️ Row count mismatch in {repo} - {file_type}. Skipping.")
            continue

        texts = [df[text_col].astype(str).tolist() for df in dfs.values()]
        if not all(texts[0] == t for t in texts[1:]):
            warnings.warn(f"    ⚠️ Text column mismatch across models in {repo} - {file_type}. Skipping.")
            continue

        base_df = dfs[model_names[0]].copy()

        for model in model_names:
            base_df[f"{model}_sentiment_label"] = dfs[model][f"{model}_sentiment_label"]
            base_df[f"{model}_confidence"] = dfs[model][f"{model}_confidence"]

        print(f"    🔀 Applying majority voting...")
        vote_results = base_df.apply(compute_majority_vote, axis=1)
        base_df = pd.concat([base_df, vote_results], axis=1)

        # Keep only specified columns + model results + final decision columns
        result_cols = keep_cols + \
                      [f"{model}_sentiment_label" for model in model_names] + \
                      [f"{model}_confidence" for model in model_names] + \
                      ["majority_label", "final_decision", "decision_reason"]

        output_df = base_df[result_cols].copy()

        output_dir = os.path.join(MERGED_OUTPUT_DIR, owner, repo_name)
        os.makedirs(output_dir, exist_ok=True)

        output_file = os.path.join(output_dir, f"{base_name}_classified.csv")
        output_df.to_csv(output_file, index=False)
        print(f"    💾 Saved: {output_file}")

print("\n✅ Merging and majority voting process completed!")