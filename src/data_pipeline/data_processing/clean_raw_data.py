import os
import pandas as pd
import re
import unicodedata
import emoji
from src.config import CONFIG
from bs4 import BeautifulSoup

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
REVIEW_COMMENTS_FILE = CONFIG['github']['file_names']['review_comments']
SUMMARY_FILE = CONFIG['github']['file_names']['summary_data']

# Patterns
RETEST_PATTERNS = re.compile(r"^\s*(retest\s*this\s*please|please\s*retest|retest\s*this|retest)\s*$", re.IGNORECASE)
BOT_USER_PATTERN = re.compile(r"(bot|ci[-]?cd|automation|github[-]?actions|travis|circleci|jenkins|codecov|dependabot|asfgit)", re.IGNORECASE)
BOT_TRIGGER_PATTERNS = re.compile(r"^\s*(retest(\s*this\s*please)?|/test\s+[\w-]+|/retest|/hold|/unhold|/ok-to-test|/assign(\s+[-]?[\w-]+)?|/unassign|/remove-sig\s+[\w-]+|/auto-cc(\s+[-]?[\w-]+)?)\s*$", re.IGNORECASE)
BUILD_RESULTS_PATTERN = re.compile(r".*refer to this link for build results.*", re.IGNORECASE)
K8S_TEST_PATTERN = re.compile(r"^\s*/test\s+pull-kubernetes-[\w-]+\s*$", re.IGNORECASE)
SPECIFIC_COMMANDS_PATTERN = re.compile(r"^\s*(-bot|/(assign|test|remove-sig|remove-wg|remove-[\w-]+|unassign|shrug|skip)).*", re.IGNORECASE)

def remove_html_tags(text):
    if pd.isna(text):
        return ""
    return BeautifulSoup(text, "html.parser").get_text(separator=" ", strip=True)

def is_bot_user(username):
    return bool(BOT_USER_PATTERN.search(username)) if pd.notna(username) else False

def is_bot_trigger(text):
    return bool(BOT_TRIGGER_PATTERNS.match(text)) if pd.notna(text) else False

def is_build_results_link(text):
    return bool(BUILD_RESULTS_PATTERN.match(text)) if pd.notna(text) else False

def has_bot_mention(text):
    if pd.isna(text):
        return False
    text = text.strip().lower()
    if text.startswith("-bot"):
        return True
    mentions = re.findall(r"@(\w+)", text)
    return any(is_bot_user(mention) for mention in mentions)

def remove_mentions(text):
    return re.sub(r"@\w+", "", text)

def flag_bot_related(df):
    df["is_bot_generated"] = df["user"].apply(is_bot_user)
    df["is_bot_trigger"] = df["body"].apply(is_bot_trigger)
    df["is_build_results_link"] = df["body"].apply(is_build_results_link)
    df["is_bot_mention"] = df["body"].apply(has_bot_mention)
    df["is_k8s_test_command"] = df["body"].apply(
        lambda text: bool(K8S_TEST_PATTERN.match(text.strip())) if pd.notna(text) else False
    )
    df["is_specific_command"] = df["body"].apply(
        lambda text: bool(SPECIFIC_COMMANDS_PATTERN.match(text.strip())) if pd.notna(text) else False
    )

    df["is_bot_related"] = df[
        [
            "is_bot_generated", "is_bot_trigger", "is_build_results_link",
            "is_bot_mention", "is_k8s_test_command", "is_specific_command"
        ]
    ].any(axis=1)

    flagged_df = df[df["is_bot_related"]].copy()
    cleaned_df = df[~df["is_bot_related"]].copy()
    return flagged_df, cleaned_df

def replace_urls(text):
    if pd.isna(text):
        return ""
    text = re.sub(r'\[.*?\]\((https?://[^\s]+)\)', '[a link]', text)
    text = re.sub(r'https?://\S+', '[a link]', text)
    return text

def remove_quotes(text):
    if pd.isna(text):
        return ""
    return "\n".join(line for line in str(text).split("\n") if not line.strip().startswith(">"))

def clean_whitespace(text):
    return re.sub(r"\s+", " ", text).strip()

def replace_code_blocks(text):
    text = re.sub(r"```.*?```", "[code block]", text, flags=re.DOTALL)
    return re.sub(r"^(\s{4,}|\t+).*", "[code block]", text, flags=re.MULTILINE)

def replace_emojis(text):
    return emoji.demojize(text, delimiters=(":", ":"))

def remove_symbol_spam(text):
    return re.sub(r"([!?])\1{2,}", r"\1", text)

def enforce_utf8(text):
    return text.encode("utf-8", "ignore").decode("utf-8") if pd.notna(text) else text

def normalize_unicode(text):
    return unicodedata.normalize("NFKC", text) if pd.notna(text) else text

def clean_repo_data(repo):
    print(f"Cleaning data for {repo}...")
    repo_org, repo_name = repo.split("/")
    base_path = os.path.join(RAW_DIR, repo_org, repo_name)

    prs = pd.read_csv(os.path.join(base_path, PULL_REQUESTS_FILE))
    comments = pd.read_csv(os.path.join(base_path, COMMENTS_FILE))
    reviews = pd.read_csv(os.path.join(base_path, REVIEWS_FILE))
    review_comments = pd.read_csv(os.path.join(base_path, REVIEW_COMMENTS_FILE))

    flagged_comments, comments = flag_bot_related(comments)
    flagged_review_comments, review_comments = flag_bot_related(review_comments)

    for df in [comments, reviews, review_comments]:
        df["body"] = (df["body"]
            .apply(lambda x: remove_html_tags(x) if pd.notna(x) else "")
            .apply(remove_quotes)
            .apply(remove_mentions)
            .apply(replace_urls)
            .apply(clean_whitespace)
            .apply(replace_code_blocks)
            .apply(replace_emojis)
            .str.lower()
            .apply(remove_symbol_spam)
            .apply(enforce_utf8)
            .apply(normalize_unicode)
        )

    for df in [comments, reviews, review_comments]:
        df["is_only_link"] = df["body"].apply(
            lambda x: bool(re.fullmatch(r"\s*\[a link\](?:\s*\[a link\]\s*)*\s*", str(x).strip()))
        )
        df["is_only_code_block"] = df["body"].apply(
            lambda x: bool(re.fullmatch(r"\s*\[code block\](?:\s*\[code block\]\s*)*\s*", str(x).strip()))
        )
        df.drop(df[df["is_only_link"] | df["is_only_code_block"]].index, inplace=True)
        df.drop(columns=["is_only_link", "is_only_code_block"], inplace=True)
        df.dropna(subset=["body"], inplace=True)
        df.drop(df[df["body"].str.strip() == ""].index, inplace=True)

    # Drop bot flags before saving cleaned files
    bot_cols = [
        "is_bot_generated", "is_bot_trigger", "is_build_results_link",
        "is_bot_mention", "is_k8s_test_command", "is_specific_command", "is_bot_related"
    ]
    comments.drop(columns=[col for col in bot_cols if col in comments.columns], inplace=True)
    review_comments.drop(columns=[col for col in bot_cols if col in review_comments.columns], inplace=True)

    save_path = os.path.join(PROCESSED_DIR, repo_org, repo_name)
    extra_path = os.path.join(save_path, "extra")
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(extra_path, exist_ok=True)

    # Save primary cleaned files to root
    prs.to_csv(os.path.join(save_path, PULL_REQUESTS_FILE.replace(".csv", "_clean.csv")), index=False)
    comments.to_csv(os.path.join(save_path, COMMENTS_FILE.replace(".csv", "_clean.csv")), index=False)
    reviews.to_csv(os.path.join(save_path, REVIEWS_FILE.replace(".csv", "_clean.csv")), index=False)
    review_comments.to_csv(os.path.join(save_path, REVIEW_COMMENTS_FILE.replace(".csv", "_clean.csv")), index=False)

    # Save bot-related files ONLY in "extra" directory
    flagged_comments.to_csv(os.path.join(extra_path, "bot_comments.csv"), index=False)
    flagged_review_comments.to_csv(os.path.join(extra_path, "bot_review_comments.csv"), index=False)

    print(f"  Original comments: {len(comments) + len(flagged_comments)}, Cleaned: {len(comments)}, Bot-related: {len(flagged_comments)}")
    print(f"  Original review comments: {len(review_comments) + len(flagged_review_comments)}, Cleaned: {len(review_comments)}, Bot-related: {len(flagged_review_comments)}")
    print(f"  Original reviews: {len(reviews)}, Cleaned: {len(reviews)}")

for repo in repositories:
    clean_repo_data(repo)

print("✅ Data cleaning completed!")
