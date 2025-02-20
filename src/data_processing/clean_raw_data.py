import os
import pandas as pd
import re
import unicodedata
import emoji
from src.config import CONFIG

# Load directories from config
RAW_DIR = CONFIG['github']['raw_dir']
PROCESSED_DIR = CONFIG['github']['processed_dir']
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Load repositories from config
repositories = CONFIG['github']['repositories']

# Load file names from config
PULL_REQUESTS_FILE = CONFIG['github']['file_names']['pull_requests']
COMMENTS_FILE = CONFIG['github']['file_names']['comments']
REVIEWS_FILE = CONFIG['github']['file_names']['reviews']
RETEST_PATTERNS = re.compile(r"^\s*(retest\s*this\s*please|please\s*retest|retest\s*this|retest)\s*$", re.IGNORECASE)
# messages to trigger the automation bots:
    # failure unrelated retest this please
    # retest this please
    # /test pull-kubernetes-e2e-kind-canary
    # /test pull-kubernetes-e2e-kind-ipv6
    # /assign rebased
    # /unassign
    # 




#  Function to clean data
def clean_repo_data(repo):
    print(f"Cleaning data for {repo}...")

    repo_org, repo_name = repo.split("/")
    repo_path = os.path.join(RAW_DIR, repo_org, repo_name)
    
    pr_file = os.path.join(repo_path, PULL_REQUESTS_FILE)
    comments_file = os.path.join(repo_path, COMMENTS_FILE)
    reviews_file = os.path.join(repo_path, REVIEWS_FILE)

    if not (os.path.exists(pr_file) and os.path.exists(comments_file) and os.path.exists(reviews_file)):
        print(f"Skipping {repo}: Missing files")
        return

    prs = pd.read_csv(pr_file)
    comments = pd.read_csv(comments_file)
    reviews = pd.read_csv(reviews_file)

    # Remove bot-generated comments
    comments = comments[~comments["user"].str.contains("bot|ci|automation|asfgit", case=False, na=False)]

    # Remove empty review comments
    reviews = reviews[reviews["body"].notna() & (reviews["body"].str.strip() != "")]

    # Remove quoted text (lines starting with ">")
    def remove_quotes(text):
        if pd.isna(text):
            return text
        return "\n".join([line for line in text.split("\n") if not line.strip().startswith(">")])

    comments["body"] = comments["body"].apply(remove_quotes)
    reviews["body"] = reviews["body"].apply(remove_quotes)

    # Remove comments that only contain "retest this please" (or similar)
    comments = comments[~comments["body"].str.match(RETEST_PATTERNS, na=False)]

    # Remove user mentions
    def remove_mentions(text):
        return re.sub(r"@\w+", "", text)

    comments["body"] = comments["body"].apply(remove_mentions)

    # Replace URLs with "a link"
    def replace_urls(text):
        return re.sub(r"https?://\S+", "a link", text)

    comments["body"] = comments["body"].apply(replace_urls)

    # Normalize whitespace (remove extra spaces and newlines)
    def clean_whitespace(text):
        return re.sub(r"\s+", " ", text).strip()

    comments["body"] = comments["body"].apply(clean_whitespace)

    # Remove full code blocks enclosed in triple backticks (```)
    def replace_code_blocks(text):
        return re.sub(r"```.*?```", "[code block]", text, flags=re.DOTALL)

    comments["body"] = comments["body"].apply(replace_code_blocks)

    # Convert emojis to GitHub-style shortcodes (e.g., 🔥 → :fire:)
    def replace_emojis(text):
        return emoji.demojize(text, delimiters=(":", ":"))

    comments["body"] = comments["body"].apply(replace_emojis)
    reviews["body"] = reviews["body"].apply(replace_emojis)

    # Convert to lowercase
    comments["body"] = comments["body"].str.lower()

    # Handle unnecessary symbol repetition (e.g., "!!!", "???" → "!", "?")
    def remove_symbol_spam(text):
        return re.sub(r"([!?])\1{2,}", r"\1", text)  # Replace repeated `!!!` or `???` with `!` or `?`

    comments["body"] = comments["body"].apply(remove_symbol_spam)

    # Handle Non-ASCII / Special Encodings (enforce UTF-8)
    def enforce_utf8(text):
        return text.encode("utf-8", "ignore").decode("utf-8")

    comments["body"] = comments["body"].apply(enforce_utf8)

    # Handle Emojis and Special Characters (normalize Unicode)
    def normalize_unicode(text):
        return unicodedata.normalize("NFKC", text)  # Normalize characters

    comments["body"] = comments["body"].apply(normalize_unicode)


    # Remove comments with empty body
    comments = comments[comments["body"].str.strip() != ""]

    # Keep only PRs with at least one valid comment or review
    valid_prs = set(comments["pr_number"]).union(set(reviews["pr_number"]))
    prs = prs[prs["number"].isin(valid_prs)]

    # Save cleaned data
    repo_processed_path = os.path.join(PROCESSED_DIR, repo_org, repo_name)
    os.makedirs(repo_processed_path, exist_ok=True)

    prs.to_csv(os.path.join(repo_processed_path, PULL_REQUESTS_FILE.replace(".csv", "_clean.csv")), index=False)
    comments.to_csv(os.path.join(repo_processed_path, COMMENTS_FILE.replace(".csv", "_clean.csv")), index=False)
    reviews.to_csv(os.path.join(repo_processed_path, REVIEWS_FILE.replace(".csv", "_clean.csv")), index=False)

    print(f"Cleaned data saved for {repo}.")

# Process all repositories
for repo in repositories:
    clean_repo_data(repo)

print("✅ Data cleaning completed!")
