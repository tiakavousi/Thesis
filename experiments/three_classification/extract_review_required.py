import os
import pandas as pd
from src.config import CONFIG

MERGED_DIR = "data/merged_voting"
OUTPUT_DIR = "data/review_required"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FILE_TYPES = {
    "comments_classified.csv": "comments",
    "review_comments_classified.csv": "review_comments",
    "reviews_classified.csv": "reviews",
    "pull_requests_classified.csv": "pull_requests"
}

repositories = CONFIG["github"]["repositories"]

general_records = []
pr_records = []

for repo in repositories:
    owner, repo_name = repo.split("/")
    repo_path = os.path.join(MERGED_DIR, owner, repo_name)

    for file_name, file_type in FILE_TYPES.items():
        file_path = os.path.join(repo_path, file_name)
        if not os.path.exists(file_path):
            continue

        df = pd.read_csv(file_path)
        df = df[df["final_decision"] == "review"]

        if df.empty:
            continue

        if file_type == "pull_requests":
            for _, row in df.iterrows():
                pr_records.append({
                    "repository": f"{owner}/{repo_name}",
                    "file_type": file_type,
                    "id": row.get("id"),
                    "pr_number": row.get("number"),
                    "text": row.get("title", ""),
                    "majority_label": row.get("majority_label"),
                    "final_decision": row.get("final_decision"),
                    "decision_reason": row.get("decision_reason")
                })
        else:
            for _, row in df.iterrows():
                general_records.append({
                    "repository": f"{owner}/{repo_name}",
                    "file_type": file_type,
                    "id": row.get("id"),
                    "pr_number": row.get("pr_number"),
                    "text": row.get("body", ""),
                    "majority_label": row.get("majority_label"),
                    "final_decision": row.get("final_decision"),
                    "decision_reason": row.get("decision_reason")
                })

# Save outputs
if general_records:
    pd.DataFrame(general_records).to_csv(
        os.path.join(OUTPUT_DIR, "comments_reviews.csv"), index=False
    )
if pr_records:
    pd.DataFrame(pr_records).to_csv(
        os.path.join(OUTPUT_DIR, "pull_requests.csv"), index=False
    )

print("✅ Review-required entries extracted and saved.")
