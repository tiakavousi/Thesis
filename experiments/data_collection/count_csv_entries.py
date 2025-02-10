import os
import csv
from src.config import CONFIG

def count_csv_entries(filepath):
    """
    Counts the number of records in a CSV file (excluding the header).
    Returns 0 if the file does not exist.
    """
    if not os.path.exists(filepath):
        return 0
    
    with open(filepath, mode='r', encoding='utf-8') as file:
        return sum(1 for _ in csv.DictReader(file))

def main():
    # List of repositories to check.
    repositories = CONFIG['github']['repositories']
    raw_dir = CONFIG['github']['raw_dir']
    pr_file_name = CONFIG['github']['file_names']['pull_requests']
    comments_file_name = CONFIG['github']['file_names']['comments']
    reviews_file_name = CONFIG['github']['file_names']['reviews']
    total_prs = 0
    total_comments = 0
    total_reviews = 0
    
    print("Summary of collected data:")
    print("=" * 50)
    
    for repo in repositories:
        # Construct the directory where the repo's CSV files are stored.
        repo_dir = os.path.join(raw_dir, repo)
        pr_path = os.path.join(repo_dir, pr_file_name)
        comments_path = os.path.join(repo_dir, comments_file_name)
        reviews_path = os.path.join(repo_dir, reviews_file_name)
        
        pr_count = count_csv_entries(pr_path)
        comment_count = count_csv_entries(comments_path)
        review_count = count_csv_entries(reviews_path)
        
        total_prs += pr_count
        total_comments += comment_count
        total_reviews += review_count
        
        print(f"Repository: {repo}")
        print(f"  Pull Requests: {pr_count}")
        print(f"  Comments:      {comment_count}")
        print(f"  Reviews:       {review_count}")
        print("-" * 50)
    
    print("Overall Summary:")
    print("=" * 50)
    print(f"Total Pull Requests: {total_prs}")
    print(f"Total Comments:      {total_comments}")
    print(f"Total Reviews:       {total_reviews}")
    print("=" * 50)

if __name__ == "__main__":
    main()
