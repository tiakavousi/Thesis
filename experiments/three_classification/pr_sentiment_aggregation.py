import os
import pandas as pd
from src.config import CONFIG

# --- Configuration ---
WEIGHTS = {
    "title": 0.5,
    "comments": 1.0,
    "reviews": 2.0,
    "review_comments": 3.0,
}

REPOSITORIES = CONFIG["github"]["repositories"]

FILE_NAMES = {
    "pull_requests": "pull_requests_classified.csv",
    "comments": "comments_classified.csv",
    "reviews": "reviews_classified.csv",
    "review_comments": "review_comments_classified.csv",
}

INPUT_BASE_DIR = "data/labeled_with_manual"
OUTPUT_FILE = "data/pr_sentiment_aggregated/pr_sentiment_summary.csv"

# Ensure output directory exists
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

# --- Helpers ---

def get_final_label(row):
    """Use manual_label unless it's 2 (fallback to majority_label)."""
    return row["manual_label"] if row["manual_label"] != 2 else row["majority_label"]

def count_sentiments(df):
    """Count positive, neutral, negative labels based on final_label column."""
    return (
        (df["final_label"] == 1).sum(),
        (df["final_label"] == 0).sum(),
        (df["final_label"] == -1).sum()
    )

# --- Main Processing ---

pr_summary_rows = []

for repo_full in REPOSITORIES:
    repo_name = repo_full.split("/")[-1]
    repo_path = os.path.join(INPUT_BASE_DIR, *repo_full.split("/"))
    print(f"\n📦 Processing repository: {repo_full}")

    try:
        df_title = pd.read_csv(os.path.join(repo_path, FILE_NAMES["pull_requests"]))
        df_comments = pd.read_csv(os.path.join(repo_path, FILE_NAMES["comments"]))
        df_reviews = pd.read_csv(os.path.join(repo_path, FILE_NAMES["reviews"]))
        df_review_comments = pd.read_csv(os.path.join(repo_path, FILE_NAMES["review_comments"]))
    except FileNotFoundError:
        print(f"⚠️ Missing one or more files in {repo_path}, skipping this repo.")
        continue

    # Assign final labels
    for df in [df_comments, df_reviews, df_review_comments]:
        df["final_label"] = df.apply(get_final_label, axis=1)
    df_title["final_label"] = df_title.apply(get_final_label, axis=1)

    for _, pr in df_title.iterrows():
        pr_number = pr["number"]
        
        merged_at = str(pr.get("merged_at")).strip().lower()
        pr_outcome = "merged" if merged_at and merged_at != "nan" else "rejected"

        # # Filter components by PR number
        comments = df_comments[df_comments["pr_number"] == pr_number]
        reviews = df_reviews[df_reviews["pr_number"] == pr_number]
        review_comments = df_review_comments[df_review_comments["pr_number"] == pr_number]

        # Count sentiment labels per component
        pos_c, neu_c, neg_c = count_sentiments(comments)
        pos_r, neu_r, neg_r = count_sentiments(reviews)
        pos_rc, neu_rc, neg_rc = count_sentiments(review_comments)
        neg_t = 1 if pr["final_label"] == -1 else 0

        # Weighted sentiment score
        weighted_neg = (
            WEIGHTS["comments"] * neg_c +
            WEIGHTS["reviews"] * neg_r +
            WEIGHTS["review_comments"] * neg_rc +
            WEIGHTS["title"] * neg_t
        )
        weighted_total = (
            WEIGHTS["comments"] * len(comments) +
            WEIGHTS["reviews"] * len(reviews) +
            WEIGHTS["review_comments"] * len(review_comments) +
            WEIGHTS["title"] * 1
        )

        # Final metrics
        weighted_negativity_ratio = weighted_neg / weighted_total if weighted_total > 0 else 0
        has_any_negative = 1 if weighted_neg > 0 else 0

        # Store results
        pr_summary_rows.append({
            "repository": repo_name,
            "pr_number": pr_number,
            "pr_outcome": pr_outcome,
            "weighted_negativity_ratio": weighted_negativity_ratio,
            "total_feedback": weighted_total,
            "has_any_negative_feedback": has_any_negative,
            "num_positive_comments": pos_c,
            "num_neutral_comments": neu_c,
            "num_negative_comments": neg_c,
            "total_comments": len(comments),
            "num_positive_reviews": pos_r,
            "num_neutral_reviews": neu_r,
            "num_negative_reviews": neg_r,
            "total_reviews": len(reviews),
            "num_positive_review_comments": pos_rc,
            "num_neutral_review_comments": neu_rc,
            "num_negative_review_comments": neg_rc,
            "total_review_comments": len(review_comments),
        })

# Save results
summary_df = pd.DataFrame(pr_summary_rows)
summary_df.to_csv(OUTPUT_FILE, index=False)
print(f"\n✅ Saved PR sentiment summary to {OUTPUT_FILE}")
