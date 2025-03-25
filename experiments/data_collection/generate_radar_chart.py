import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from math import pi

# Add project root to import config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.config import CONFIG

# Setup
input_root = "/Users/tayebekavousi/Desktop/github_sa/data/raw"
output_dir = "./raw_data_exploration"
os.makedirs(output_dir, exist_ok=True)

radar_data = []

print("📊 Generating radar chart from repository data...\n")

for repo_path in CONFIG["github"]["repositories"]:
    org, repo = repo_path.split("/")
    full_path = os.path.join(input_root, org, repo)
    full_repo_name = f"{org}/{repo}"
    try:
        pr_df = pd.read_csv(os.path.join(full_path, CONFIG["github"]["file_names"]["pull_requests"]))
        comments_df = pd.read_csv(os.path.join(full_path, CONFIG["github"]["file_names"]["comments"]))
        reviews_df = pd.read_csv(os.path.join(full_path, CONFIG["github"]["file_names"]["reviews"]))
        review_comments_df = pd.read_csv(os.path.join(full_path, CONFIG["github"]["file_names"]["review_comments"]))

        pr_comment_counts = comments_df['pr_number'].value_counts()
        review_counts = reviews_df['pr_number'].value_counts()
        review_comment_counts = review_comments_df['pr_number'].value_counts()

        radar_data.append({
            'repo': full_repo_name,
            'avg_comments_per_pr': pr_comment_counts.mean(),
            'avg_reviews_per_pr': review_counts.mean(),
            'avg_review_comments_per_pr': review_comment_counts.mean()
        })

    except Exception as e:
        print(f"❌ Skipping {full_repo_name} due to error: {e}")

# === Plot Radar Chart ===
try:
    print("\n🧭 Creating radar chart with sorted legend...")

    radar_df = pd.DataFrame(radar_data)
    radar_df["total_avg_communication"] = (
        radar_df["avg_comments_per_pr"] +
        radar_df["avg_reviews_per_pr"] +
        radar_df["avg_review_comments_per_pr"]
    )
    radar_df = radar_df.sort_values("total_avg_communication", ascending=False)
    radar_df = radar_df.drop(columns="total_avg_communication").set_index("repo")

    categories = radar_df.columns.tolist()
    N = len(categories)
    angles = [n / float(N) * 2 * pi for n in range(N)] + [0]

    plt.figure(figsize=(13, 13))
    color_map = plt.cm.get_cmap('tab20', len(radar_df))

    for i, (label, row) in enumerate(radar_df.iterrows()):
        values = row.tolist() + [row.tolist()[0]]
        plt.polar(angles, values, label=label, alpha=0.5, linewidth=2, color=color_map(i))

    plt.xticks(angles[:-1], categories, fontsize=12)
    plt.title("Repository Comparison Summary (Radar Chart)", fontsize=16, pad=20)
    plt.legend(
        loc='center left',
        bbox_to_anchor=(1.15, 0.5),
        fontsize=10,
        title="Repositories",
        labels=radar_df.index.tolist()
    )
    plt.tight_layout()
    output_path = os.path.join(output_dir, "comparison_summary_radar.pdf")
    plt.savefig(output_path)
    plt.close()
    print(f"✅ Radar chart saved to {output_path}")

except Exception as e:
    print(f"❌ Failed to generate radar chart: {e}")
