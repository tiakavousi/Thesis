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
REVIEW_COMMENTS_FILE = CONFIG['github']['file_names']['review_comments']
SUMMARY_FILE = CONFIG['github']['file_names']['summary_data']

# Existing pattern for retest messages (for comments that solely say "retest this please")
RETEST_PATTERNS = re.compile(
    r"^\s*(retest\s*this\s*please|please\s*retest|retest\s*this|retest)\s*$",
    re.IGNORECASE
)

# --- Extended Patterns for Bot Detection ---
BOT_USER_PATTERN = re.compile(
    r"(bot|ci[-]?cd|automation|github[-]?actions|travis|circleci|jenkins|codecov|dependabot)",
    re.IGNORECASE
)

# Extended BOT_TRIGGER_PATTERNS to cover additional bot commands:
BOT_TRIGGER_PATTERNS = re.compile(
    r"^\s*(retest(\s*this\s*please)?|/test\s+[\w-]+|/retest|/hold|/unhold|/ok-to-test|/assign(\s+[-]?[\w-]+)?|/unassign|/remove-sig\s+[\w-]+|/auto-cc(\s+[-]?[\w-]+)?)\s*$",
    re.IGNORECASE
)

# New pattern for build results links - stricter pattern to match the exact phrase
BUILD_RESULTS_PATTERN = re.compile(
    r".*refer to this link for build results.*",
    re.IGNORECASE
)

def is_bot_user(username):
    """Return True if the username indicates a CI/CD bot."""
    return bool(BOT_USER_PATTERN.search(username)) if pd.notna(username) else False

def is_bot_trigger(text):
    """Return True if the comment text matches a known bot trigger command."""
    return bool(BOT_TRIGGER_PATTERNS.match(text)) if pd.notna(text) else False

def is_build_results_link(text):
    """Return True if the comment contains a build results link reference."""
    return bool(BUILD_RESULTS_PATTERN.match(text)) if pd.notna(text) else False

def has_bot_mention(text):
    """
    Return True if the text either contains an '@' mention of a bot
    or starts with a dash-based bot mention (e.g. "-bot").
    """
    if pd.isna(text):
        return False
    # Convert to string, strip leading/trailing whitespace and lower-case it
    text_stripped = str(text).strip().lower()
    # Remove the comment if it starts with "-bot"
    if text_stripped.startswith("-bot"):
        return True
    # Also check for '@' mentions of bots
    mentions = re.findall(r"@(\w+)", text_stripped)
    if any(is_bot_user(mention) for mention in mentions):
        return True
    return False

def remove_mentions(text):
    """Remove all user mentions from the text."""
    return re.sub(r"@\w+", "", text)

# Function to clean data for each repository
def clean_repo_data(repo):
    print(f"Cleaning data for {repo}...")
    repo_org, repo_name = repo.split("/")

    repo_path = os.path.join(RAW_DIR, repo_org, repo_name)
    pr_file = os.path.join(repo_path, PULL_REQUESTS_FILE)
    comments_file = os.path.join(repo_path, COMMENTS_FILE)
    reviews_file = os.path.join(repo_path, REVIEWS_FILE)
    review_comments_file = os.path.join(repo_path, REVIEW_COMMENTS_FILE)

    prs = pd.read_csv(pr_file)
    comments = pd.read_csv(comments_file)
    reviews = pd.read_csv(reviews_file)
    review_comments = pd.read_csv(review_comments_file)

    # --- Flag Bot-Related Comments ---
    comments["is_bot_generated"] = comments["user"].apply(is_bot_user)
    comments["is_bot_trigger"] = comments["body"].apply(is_bot_trigger)
    comments["is_build_results_link"] = comments["body"].apply(is_build_results_link)

    total_comments = len(comments)
    bot_generated_count = comments["is_bot_generated"].sum()
    bot_trigger_count = comments["is_bot_trigger"].sum()
    build_results_count = comments["is_build_results_link"].sum()

    # Flagged (bot-related) comments before removal
    flagged_comments = comments[comments["is_bot_generated"] | comments["is_bot_trigger"] | comments["is_build_results_link"]].copy()
    removed_flagged_count = len(flagged_comments)
    remaining_after_flag = total_comments - removed_flagged_count

    print(f"Total comments: {total_comments}")
    print(f"Bot-generated comments: {bot_generated_count}, Bot trigger comments: {bot_trigger_count}")
    print(f"Build results link comments: {build_results_count}")
    print(f"Removing {removed_flagged_count} flagged comments; {remaining_after_flag} comments remain.")

    # Create processed folder for repo
    repo_processed_path = os.path.join(PROCESSED_DIR, repo_org, repo_name)
    os.makedirs(repo_processed_path, exist_ok=True)

    # Save flagged bot comments to a separate file
    flagged_file_path = os.path.join(repo_processed_path, "bot_comments.csv")
    flagged_comments.to_csv(flagged_file_path, index=False)

    # Remove flagged comments from the main DataFrame
    comments = comments[~(comments["is_bot_generated"] | comments["is_bot_trigger"] | comments["is_build_results_link"])]

    # --- Additional Bot Mention Check ---
    # Remove comments that mention a bot (using '@' or starting with a dash-based bot mention)
    initial_count = len(comments)
    comments = comments[~comments["body"].apply(has_bot_mention)]
    bot_mention_removed_count = initial_count - len(comments)
    print(f"Removed an additional {bot_mention_removed_count} comments due to bot mentions.")
    
    # Double-check for build results links again after all cleaning
    comments = comments[~comments["body"].str.contains("refer to this link for build results", case=False, na=False)]
    review_comments = review_comments[~review_comments["body"].str.contains("refer to this link for build results", case=False, na=False)]

    # --- Process Review Comments similar to regular Comments ---
    review_comments["is_bot_generated"] = review_comments["user"].apply(is_bot_user)
    review_comments["is_bot_trigger"] = review_comments["body"].apply(is_bot_trigger)
    review_comments["is_build_results_link"] = review_comments["body"].apply(is_build_results_link)
    
    total_review_comments = len(review_comments)
    rc_bot_generated_count = review_comments["is_bot_generated"].sum()
    rc_bot_trigger_count = review_comments["is_bot_trigger"].sum()
    rc_build_results_count = review_comments["is_build_results_link"].sum()
    
    # Flagged review comments before removal
    flagged_review_comments = review_comments[
        review_comments["is_bot_generated"] | 
        review_comments["is_bot_trigger"] | 
        review_comments["is_build_results_link"]
    ].copy()
    removed_rc_flagged_count = len(flagged_review_comments)
    remaining_rc_after_flag = total_review_comments - removed_rc_flagged_count
    
    print(f"Total review comments: {total_review_comments}")
    print(f"Bot-generated review comments: {rc_bot_generated_count}, Bot trigger review comments: {rc_bot_trigger_count}")
    print(f"Build results link review comments: {rc_build_results_count}")
    print(f"Removing {removed_rc_flagged_count} flagged review comments; {remaining_rc_after_flag} review comments remain.")
    
    # Save flagged bot review comments to a separate file
    flagged_rc_file_path = os.path.join(repo_processed_path, "bot_review_comments.csv")
    flagged_review_comments.to_csv(flagged_rc_file_path, index=False)
    
    # Remove flagged review comments from the main DataFrame
    review_comments = review_comments[
        ~(review_comments["is_bot_generated"] | 
          review_comments["is_bot_trigger"] | 
          review_comments["is_build_results_link"])
    ]
    
    # Remove review comments that mention a bot
    initial_rc_count = len(review_comments)
    review_comments = review_comments[~review_comments["body"].apply(has_bot_mention)]
    rc_bot_mention_removed_count = initial_rc_count - len(review_comments)
    print(f"Removed an additional {rc_bot_mention_removed_count} review comments due to bot mentions.")
    
    # --- Data Cleaning Steps for Comments & Reviews ---
    # Remove empty review comments and review comment bodies
    reviews = reviews[reviews["body"].notna() & (reviews["body"].str.strip() != "")]
    review_comments = review_comments[review_comments["body"].notna() & (review_comments["body"].str.strip() != "")]

    # Remove quoted text (lines starting with ">")
    def remove_quotes(text):
        if pd.isna(text):
            return text
        return "\n".join([line for line in text.split("\n") if not line.strip().startswith(">")])
    comments["body"] = comments["body"].apply(remove_quotes)
    reviews["body"] = reviews["body"].apply(remove_quotes)
    review_comments["body"] = review_comments["body"].apply(remove_quotes)

    # Remove comments that only contain "retest this please" (or similar)
    comments = comments[~comments["body"].str.match(RETEST_PATTERNS, na=False)]
    review_comments = review_comments[~review_comments["body"].str.match(RETEST_PATTERNS, na=False)]

    # Remove remaining user mentions (safe now since comments with bot mentions were removed)
    comments["body"] = comments["body"].apply(remove_mentions)
    review_comments["body"] = review_comments["body"].apply(remove_mentions)

    # Replace URLs with "a link"
    def replace_urls(text):
        return re.sub(r"https?://\S+", "[a link]", text)
    comments["body"] = comments["body"].apply(replace_urls)
    review_comments["body"] = review_comments["body"].apply(replace_urls)

    # Normalize whitespace
    def clean_whitespace(text):
        return re.sub(r"\s+", " ", text).strip()
    comments["body"] = comments["body"].apply(clean_whitespace)
    reviews["body"] = reviews["body"].apply(clean_whitespace)
    review_comments["body"] = review_comments["body"].apply(clean_whitespace)

    # Remove full code blocks enclosed in triple backticks or with indentation
    def replace_code_blocks(text):
        if pd.isna(text):
            return text
        # Replace triple backtick code blocks
        text = re.sub(r"```.*?```", "[code block]", text, flags=re.DOTALL)
        # Replace code blocks that have 4+ spaces or tab indentation at the beginning of lines
        lines = text.split("\n")
        in_code_block = False
        code_block_lines = []
        clean_lines = []
        
        for line in lines:
            if re.match(r"^\s{4,}|^\t+", line) and line.strip():
                in_code_block = True
                code_block_lines.append(line)
            else:
                if in_code_block and len(code_block_lines) > 0:
                    # End of code block
                    clean_lines.append("[code block]")
                    code_block_lines = []
                    in_code_block = False
                if line.strip():  # Only add non-empty lines
                    clean_lines.append(line)
        
        # Handle case where code block is at the end
        if in_code_block and len(code_block_lines) > 0:
            clean_lines.append("[code block]")
            
        return "\n".join(clean_lines)
    
    comments["body"] = comments["body"].apply(replace_code_blocks)
    reviews["body"] = reviews["body"].apply(replace_code_blocks)
    review_comments["body"] = review_comments["body"].apply(replace_code_blocks)

    # Convert emojis to GitHub-style shortcodes
    def replace_emojis(text):
        return emoji.demojize(text, delimiters=(":", ":"))
    comments["body"] = comments["body"].apply(replace_emojis)
    reviews["body"] = reviews["body"].apply(replace_emojis)
    review_comments["body"] = review_comments["body"].apply(replace_emojis)

    # Convert to lowercase
    comments["body"] = comments["body"].str.lower()
    review_comments["body"] = review_comments["body"].str.lower()

    # Handle unnecessary symbol repetition
    def remove_symbol_spam(text):
        return re.sub(r"([!?])\1{2,}", r"\1", text)
    comments["body"] = comments["body"].apply(remove_symbol_spam)
    review_comments["body"] = review_comments["body"].apply(remove_symbol_spam)

    # Enforce UTF-8 encoding
    def enforce_utf8(text):
        if pd.isna(text):
            return text
        return text.encode("utf-8", "ignore").decode("utf-8")
    comments["body"] = comments["body"].apply(enforce_utf8)
    reviews["body"] = reviews["body"].apply(enforce_utf8)
    review_comments["body"] = review_comments["body"].apply(enforce_utf8)

    # Normalize Unicode (NFKC)
    def normalize_unicode(text):
        if pd.isna(text):
            return text
        return unicodedata.normalize("NFKC", text)
    comments["body"] = comments["body"].apply(normalize_unicode)
    reviews["body"] = reviews["body"].apply(normalize_unicode)
    review_comments["body"] = review_comments["body"].apply(normalize_unicode)

    # Remove comments with an empty body after cleaning
    comments = comments[comments["body"].str.strip() != ""]
    review_comments = review_comments[review_comments["body"].str.strip() != ""]

    # Keep only PRs with at least one valid comment, review, or review comment
    valid_prs = set(comments["pr_number"]).union(set(reviews["pr_number"])).union(set(review_comments["pr_number"]))
    prs = prs[prs["number"].isin(valid_prs)]

    # --- Save Cleaned Data ---
    prs.to_csv(os.path.join(repo_processed_path, PULL_REQUESTS_FILE.replace(".csv", "_clean.csv")), index=False)
    comments.to_csv(os.path.join(repo_processed_path, COMMENTS_FILE.replace(".csv", "_clean.csv")), index=False)
    reviews.to_csv(os.path.join(repo_processed_path, REVIEWS_FILE.replace(".csv", "_clean.csv")), index=False)
    review_comments.to_csv(os.path.join(repo_processed_path, REVIEW_COMMENTS_FILE.replace(".csv", "_clean.csv")), index=False)

    # --- Save Summary Information ---
    summary = pd.DataFrame([{
         "repo": repo,
         "total_comments": total_comments,
         "bot_generated": bot_generated_count,
         "bot_trigger": bot_trigger_count,
         "build_results_links": build_results_count,
         "removed_flagged_comments": removed_flagged_count,
         "removed_bot_mentions": bot_mention_removed_count,
         "remaining_comments": len(comments),
         "total_review_comments": total_review_comments,
         "rc_bot_generated": rc_bot_generated_count,
         "rc_bot_trigger": rc_bot_trigger_count,
         "rc_build_results_links": rc_build_results_count,
         "removed_flagged_rc": removed_rc_flagged_count,
         "removed_rc_bot_mentions": rc_bot_mention_removed_count,
         "remaining_review_comments": len(review_comments)
    }])
    summary_file_path = os.path.join(repo_processed_path, SUMMARY_FILE)
    summary.to_csv(summary_file_path, index=False)

    print(f"Cleaned data, bot comments, and summary saved for {repo}.")

# Process all repositories
for repo in repositories:
    clean_repo_data(repo)

print("✅ Data cleaning completed!")