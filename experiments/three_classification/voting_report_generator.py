import os
import pandas as pd
from collections import Counter
import warnings
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from src.config import CONFIG

# Directory where merged voting files are saved
MERGED_DIR = "data/merged_voting"
FILE_TYPES = [
    "comments_classified.csv",
    "review_comments_classified.csv",
    "reviews_classified.csv",
    "pull_requests_classified.csv"
]

repositories = CONFIG["github"]["repositories"]
summary_records = []

print("\n📊 Generating majority voting report...\n")

for repo in repositories:
    try:
        owner, repo_name = repo.split("/")
    except ValueError:
        warnings.warn(f"Repository format error: '{repo}'. Expected format 'owner/repo'. Skipping.")
        continue

    for file in FILE_TYPES:
        file_path = os.path.join(MERGED_DIR, owner, repo_name, file)
        if not os.path.exists(file_path):
            warnings.warn(f"Missing file: {file_path}. Skipping.")
            continue

        df = pd.read_csv(file_path)
        total = len(df)
        accepted = df[df["final_decision"] == "accept"].shape[0]
        rejected = df[df["final_decision"] == "review"].shape[0]
        reason_unanimous_low_confidence = df[(df["final_decision"] == "review") & 
                                            (df["decision_reason"] == "unanimous_low_confidence")].shape[0]
        reason_no_majority_disagreement = df[(df["final_decision"] == "review") & 
                                             (df["decision_reason"] == "no_majority_disagreement")].shape[0]
        reason_majority_low_confidence = df[(df["final_decision"] == "review") & 
                                            (df["decision_reason"] == "majority_low_confidence")].shape[0]
        
        accepted_negative = df[(df["final_decision"] == "accept") & (df["majority_label"] == -1)].shape[0]
        accepted_neutral  = df[(df["final_decision"] == "accept") & (df["majority_label"] == 0)].shape[0]
        accepted_positive = df[(df["final_decision"] == "accept") & (df["majority_label"] == 1)].shape[0]
        
        summary_records.append({
            "repository": repo,
            "file": file,
            "total_rows": total,
            "accepted": accepted,
            "rejected": rejected,
            "reason_unanimous_low_confidence": reason_unanimous_low_confidence,
            "reason_no_majority_disagreement": reason_no_majority_disagreement,
            "reason_majority_low_confidence": reason_majority_low_confidence,
            "accepted_negative": accepted_negative,
            "accepted_neutral": accepted_neutral,
            "accepted_positive": accepted_positive
        })

# Convert to DataFrame and compute totals
df_summary = pd.DataFrame(summary_records)
numeric_cols = [
    "total_rows", "accepted", "rejected",
    "reason_unanimous_low_confidence", "reason_no_majority_disagreement", "reason_majority_low_confidence",
    "accepted_negative", "accepted_neutral", "accepted_positive"
]
totals = {col: df_summary[col].sum() for col in numeric_cols}
totals["repository"] = "Total"
totals["file"] = ""
df_summary = pd.concat([df_summary, pd.DataFrame([totals])], ignore_index=True)

# Create a PDF report
os.makedirs("reports", exist_ok=True)
pdf_path = os.path.join("reports", "majority_voting_summary_report.pdf")

with PdfPages(pdf_path) as pdf:
    fig, ax = plt.subplots(figsize=(18, len(df_summary) * 0.35 + 2))
    ax.axis("off")

    table = ax.table(
        cellText=df_summary.values,
        colLabels=df_summary.columns,
        cellLoc='center',
        loc='center',
        colLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)

    ax.set_title("Majority Voting Summary Report", fontsize=14, fontweight="bold", pad=20)
    pdf.savefig(fig, bbox_inches='tight')

print(f"\n✅ PDF report saved to: {pdf_path}")
