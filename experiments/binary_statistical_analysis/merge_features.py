import pandas as pd
from pathlib import Path

# === Path Configurations ===
FEATURES_DIR = Path("/Users/tayebekavousi/Desktop/github_sa/data/features")
OUTPUT_FILE = FEATURES_DIR / "all_repo_features.csv"

# === Initialize List to Store DataFrames ===
all_dfs = []

# === Loop Through Feature Files ===
for file in FEATURES_DIR.glob("*.csv"):
    if file.name == "empty_prs.csv":
        continue  # Skip bot-related PRs file

    try:
        df = pd.read_csv(file)

        # Extract "owner/repo" from filename
        base_name = file.stem  # e.g., "apache__kafka"
        repo_name = base_name.replace("__", "/")
        df["repository"] = repo_name  # Add repo identifier

        all_dfs.append(df)
        print(f"✅ Loaded: {file.name} ({len(df)} PRs)")

    except Exception as e:
        print(f"⚠️ Failed to process {file.name}: {e}")

# === Concatenate All DataFrames ===
if all_dfs:
    combined_df = pd.concat(all_dfs, ignore_index=True)
    combined_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n🎉 Merged all features into: {OUTPUT_FILE}")
    print(f"📊 Total PRs combined: {len(combined_df)}")
else:
    print("❌ No valid feature files found to merge.")
