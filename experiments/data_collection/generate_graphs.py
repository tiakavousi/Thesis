import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

# Add project root to path so we can import config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.config import CONFIG

# Setup
sns.set(style="whitegrid", palette="Set2")
input_root = "/Users/tayebekavousi/Desktop/github_sa/data/raw"
output_dir = "./raw_data_exploration"
os.makedirs(output_dir, exist_ok=True)

summary_data = []
radar_data = []

print("Starting processing of repositories...\n")

for repo_path in CONFIG["github"]["repositories"]:
    org, repo = repo_path.split("/")
    full_path = os.path.join(input_root, org, repo)
    print(f"📂 Processing {repo_path}...")

    try:
        pr_df = pd.read_csv(os.path.join(full_path, CONFIG["github"]["file_names"]["pull_requests"]))
        comments_df = pd.read_csv(os.path.join(full_path, CONFIG["github"]["file_names"]["comments"]))
        reviews_df = pd.read_csv(os.path.join(full_path, CONFIG["github"]["file_names"]["reviews"]))
        review_comments_df = pd.read_csv(os.path.join(full_path, CONFIG["github"]["file_names"]["review_comments"]))

        # === Prepare Figures ===
        fig, axs = plt.subplots(3, 2, figsize=(16, 18))
        fig.suptitle(f"{repo} - PR Activity Summary", fontsize=18)
        axs = axs.flatten()

        # 🔢 1. Basic Overview
        counts = {
            'Pull Requests': len(pr_df),
            'Comments': len(comments_df),
            'Reviews': len(reviews_df),
            'Review Comments': len(review_comments_df)
        }
        sns.barplot(x=list(counts.keys()), y=list(counts.values()), ax=axs[0])
        axs[0].set_title("Basic Overview")
        axs[0].set_ylabel("Count")

        # 🧵 2. Comment Activity per PR
        pr_comment_counts = comments_df['pr_number'].value_counts()
        sns.violinplot(data=pr_comment_counts.values, ax=axs[1], inner="box")
        axs[1].set_title("Comment Activity per PR")
        axs[1].set_ylabel("Number of Comments")

        # 🔄 3. PR Review Behavior
        review_counts = reviews_df['pr_number'].value_counts()
        has_reviews = pr_df['number'].isin(review_counts.index)
        status = pd.Series(["With Reviews" if val else "No Reviews" for val in has_reviews])
        sns.countplot(x=status, ax=axs[2])
        axs[2].set_title("PR Review Behavior")
        axs[2].set_ylabel("Number of PRs")

        # 💬 5. Comment Density
        sns.histplot(pr_comment_counts.values, bins=20, ax=axs[3])
        axs[3].set_title("Comment Density")
        axs[3].set_xlabel("Comments per PR")

        # 🧾 7. Review vs General Comments
        gen_counts = comments_df['pr_number'].value_counts()
        rev_counts = review_comments_df['pr_number'].value_counts()
        pr_ids = list(set(gen_counts.index).union(set(rev_counts.index)))
        scatter_df = pd.DataFrame({
            'pr_number': pr_ids,
            'general_comments': [gen_counts.get(pid, 0) for pid in pr_ids],
            'review_comments': [rev_counts.get(pid, 0) for pid in pr_ids]
        })
        axs[4].scatter(scatter_df['general_comments'], scatter_df['review_comments'], alpha=0.6)
        axs[4].set_title("Review vs General Comments per PR")
        axs[4].set_xlabel("General Comments")
        axs[4].set_ylabel("Review Comments")

        # 🧭 8. Repo Summary Info (for radar)
        avg_comments = pr_comment_counts.mean()
        avg_reviews = review_counts.mean()
        avg_review_comments = review_comments_df['pr_number'].value_counts().mean()

        radar_data.append({
            'repo': repo,
            'avg_comments_per_pr': avg_comments,
            'avg_reviews_per_pr': avg_reviews,
            'avg_review_comments_per_pr': avg_review_comments
        })
        axs[5].axis('off')
        axs[5].text(0.5, 0.5, "Repo Summary collected for radar plot", ha='center', va='center', fontsize=12)

        # Save figure
        graph_path = os.path.join(output_dir, f"{repo}_graphs.pdf")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(graph_path)
        plt.close()
        print(f"✅ Saved graphs to {graph_path}")

        # Add to table
        summary_data.append([repo, len(pr_df), len(comments_df), len(reviews_df), len(review_comments_df)])

    except Exception as e:
        print(f"❌ Failed to process {repo_path}: {e}")

# === Radar Plot ===
print("\n📊 Generating radar chart for all repositories...")
try:
    radar_df = pd.DataFrame(radar_data).set_index("repo")
    categories = radar_df.columns
    N = len(categories)
    angles = [n / float(N) * 2 * pi for n in range(N)] + [0]

    plt.figure(figsize=(10, 10))
    for repo, row in radar_df.iterrows():
        values = row.tolist() + [row.tolist()[0]]
        plt.polar(angles, values, label=repo, alpha=0.3)
    plt.xticks(angles[:-1], categories)
    plt.title("Repository Comparison Summary")
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    radar_path = os.path.join(output_dir, "comparison_summary_radar.pdf")
    plt.savefig(radar_path)
    plt.close()
    print(f"✅ Radar chart saved to {radar_path}")
except Exception as e:
    print(f"❌ Radar chart generation failed: {e}")

# === Summary Table PDF ===
print("\n📄 Creating summary table PDF...")
try:
    doc = SimpleDocTemplate(os.path.join(output_dir, "summary.pdf"), pagesize=A4)
    styles = getSampleStyleSheet()
    story = [Paragraph("Repository Summary Table", styles["Heading1"]), Spacer(1, 12)]

    table_data = [["Repository", "Total PRs", "Comments", "Reviews", "Review Comments"]] + summary_data
    table = Table(table_data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#cccccc")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('FONTSIZE', (0, 0), (-1, -1), 10)
    ]))

    story.append(table)
    doc.build(story)
    print(f"✅ Summary table saved to {os.path.join(output_dir, 'summary.pdf')}")
except Exception as e:
    print(f"❌ Summary table PDF generation failed: {e}")

print("\n🎉 All done!")
