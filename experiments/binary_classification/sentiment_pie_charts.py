import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# === Paths ===
INPUT_FILE = Path("data/features/all_repo_features.csv")
OUTPUT_DIR = Path("data/features/pie_charts")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# === Load Data ===
df = pd.read_csv(INPUT_FILE)

# === Validate Required Columns ===
if "total_entries" not in df.columns or "total_negatives" not in df.columns:
    raise ValueError("Missing required columns: 'total_entries' or 'total_negatives'.")

# === Color Scheme (Dark + Light Blue) ===
colors = ["#1f4e79", "#6baed6"]  # dark blue = Positive, light blue = Negative

# === Process Each Repository ===
for repo in df["repository"].unique():
    repo_df = df[df["repository"] == repo]

    total_neg = repo_df["total_negatives"].sum()
    total_all = repo_df["total_entries"].sum()
    total_pos = total_all - total_neg

    if total_all == 0:
        print(f"⚠️ Skipping {repo}: no sentiment data.")
        continue

    # Pie Data
    values = [total_pos, total_neg]
    labels = ["Positive", "Negative"]
    percentages = [val / total_all * 100 for val in values]

    # Create Pie Chart
    fig, ax = plt.subplots(figsize=(6, 6))
    wedges, texts, autotexts = ax.pie(
        values,
        labels=None,
        colors=colors,
        autopct="%1.1f%%",
        startangle=140,
        textprops=dict(color="white", weight="bold", fontsize=12)
    )

    # Add custom sentiment labels to the slices
    for i, a in enumerate(autotexts):
        a.set_text(f"{labels[i]}\n{a.get_text()}")
        a.set_fontweight("bold")

    ax.set_title("")  # Remove title from top
    plt.axis("equal")

    # Extract repo name only (exclude owner)
    repo_name_only = repo.split("/")[-1]
    file_safe_name = repo.replace("/", "__")

    # Title below chart
    plt.figtext(0.5, 0.02, f"Sentiment Distribution – {repo_name_only}", ha="center", fontsize=12, fontweight="bold")

    # Save chart
    output_path = OUTPUT_DIR / f"{file_safe_name}_pie_chart.png"
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {output_path}")

print("🎉 All pie charts generated.")
