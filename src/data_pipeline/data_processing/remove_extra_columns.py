import pandas as pd
from pathlib import Path

# === File Paths ===
BASE_DIR = Path("data/manual_classification/qBittorrent/qBittorrent")
FILES_TO_PROCESS = {
    "comments_sentiment.csv": [
        "is_bot_generated", "is_bot_trigger", "is_build_results_link",
        "is_bot_mention", "is_k8s_test_command", "is_specific_command", "is_bot_related"
    ],
    "review_comments_sentiment.csv": [
        "is_bot_generated", "is_bot_trigger", "is_build_results_link",
        "is_bot_mention", "is_k8s_test_command", "is_specific_command", "is_bot_related"
    ]
}

# === Process Each File ===
for filename, columns_to_drop in FILES_TO_PROCESS.items():
    file_path = BASE_DIR / filename
    try:
        df = pd.read_csv(file_path)
        df = df.drop(columns=columns_to_drop, errors="ignore")
        df.to_csv(file_path, index=False)
        print(f"✅ Cleaned and saved: {filename}")
    except Exception as e:
        print(f"❌ Failed to process {filename}: {e}")
