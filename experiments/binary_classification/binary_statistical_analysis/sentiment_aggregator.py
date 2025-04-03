import os
import pandas as pd
from pathlib import Path
import sys

# Add src to path to import config
CURRENT_DIR = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_DIR.parents[2]
sys.path.append(str(PROJECT_ROOT / "src"))

from config import CONFIG

# Base directories
CLASSIFIED_BASE_DIR = Path("/Users/tayebekavousi/Desktop/github_sa/data/classified")
FEATURES_OUTPUT_DIR = Path("/Users/tayebekavousi/Desktop/github_sa/data/features")
EMPTY_PRS_FILE = FEATURES_OUTPUT_DIR / "empty_prs.csv"
FEATURES_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Prepare to track empty PRs
empty_prs = []

# Loop over each repo
for repo in CONFIG["github"]["repositories"]:
    owner, name = repo.split("/")
    repo_path = CLASSIFIED_BASE_DIR / owner / name

    try:
        pr_df = pd.read_csv(repo_path / "pull_requests_sentiment.csv")
        comments_df = pd.read_csv(repo_path / "comments_sentiment.csv")
        reviews_df = pd.read_csv(repo_path / "reviews_sentiment.csv")
        review_comments_df = pd.read_csv(repo_path / "review_comments_sentiment.csv")
    except Exception as e:
        print(f"Skipping {repo} due to missing files: {e}")
        continue

    # Add binary outcome column
    pr_df["outcome"] = pr_df["merged_at"].notnull().astype(int)

    # Helper to count negatives and totals
    def count_sentiment(df, pr_key="pr_number"):
        grouped = df.groupby(pr_key)["sentiment"].value_counts().unstack(fill_value=0)
        grouped["total"] = grouped.sum(axis=1)
        grouped["negative"] = grouped.get("Negative", 0)
        return grouped[["negative", "total"]]

    comments_stats = count_sentiment(comments_df)
    reviews_stats = count_sentiment(reviews_df)
    review_comments_stats = count_sentiment(review_comments_df)

    merged = pr_df[["number", "outcome"]].rename(columns={"number": "pr_number"})
    merged = merged.set_index("pr_number")

    merged = merged.join(comments_stats.rename(columns={"negative": "neg_comments", "total": "total_comments"}))
    merged = merged.join(reviews_stats.rename(columns={"negative": "neg_reviews", "total": "total_reviews"}))
    merged = merged.join(review_comments_stats.rename(columns={"negative": "neg_review_comments", "total": "total_review_comments"}))

    # Track and drop empty PRs
    mask_empty = merged[["total_comments", "total_reviews", "total_review_comments"]].fillna(0).sum(axis=1) == 0
    empty_prs += [(repo, pr) for pr in merged[mask_empty].index.tolist()]
    merged = merged[~mask_empty]

    # Compute overall negative ratio
    merged = merged.fillna(0)
    merged["total_negatives"] = merged["neg_comments"] + merged["neg_reviews"] + merged["neg_review_comments"]
    merged["total_entries"] = merged["total_comments"] + merged["total_reviews"] + merged["total_review_comments"]
    merged["overall_neg_ratio"] = merged["total_negatives"] / merged["total_entries"]
    merged = merged.reset_index()

    # Save feature file
    repo_safe_name = repo.replace("/", "__")
    output_path = FEATURES_OUTPUT_DIR / f"{repo_safe_name}.csv"
    merged.to_csv(output_path, index=False)
    print(f"✅ Saved features for {repo} to {output_path}")

# Save skipped PRs
empty_df = pd.DataFrame(empty_prs, columns=["repository", "pr_number"])
empty_df.to_csv(EMPTY_PRS_FILE, index=False)
print(f"📄 Saved empty PR list to {EMPTY_PRS_FILE}")
