"""
The script:

- Retrieves the 100 most-commented closed issues (excluding bots and non-PR issues).
- Fetches PR details only for these issues to ensure they are actual pull requests.
- Retrieves corresponding PR comments efficiently in batches.
- Saves the data in structured CSV files under data/REPOSITORY_NAME/.
"""

import os
import requests
import csv
import time
from config import CONFIG

# GitHub API Headers for authentication and request handling
HEADERS = {"Authorization": f"token {CONFIG['github']['token']}", "Accept": "application/vnd.github.v3+json"}
# GitHub API Rate Limit configuration
API_LIMIT = CONFIG['github']['rate_limit']['min_remaining']
# Base URL for GitHub API requests
GITHUB_API_URL = CONFIG['github']['base_url']
# Number of issues to fetch per repository
COUNT = CONFIG['github']['pull_requests']['top_n']

def check_rate_limit():
    """Checks the remaining API call limit and exits if below the threshold."""
    response = requests.get(f"{GITHUB_API_URL}/rate_limit", headers=HEADERS)
    remaining = response.json().get("rate", {}).get("remaining", 0)
    if remaining <= API_LIMIT:
        print(f"Approaching API limit. Remaining: {remaining}")
        exit()
    return remaining

def fetch_data(url):
    """Fetches data from the given GitHub API URL with rate limit checks."""
    check_rate_limit()
    response = requests.get(url, headers=HEADERS)
    if response.status_code == 200:
        return response.json()
    print(f"Error fetching {url}: {response.status_code}")
    return []

def get_most_commented_issues(repo, count=COUNT):
    """Retrieves the most commented closed issues that are pull requests."""
    url = f"{GITHUB_API_URL}/repos/{repo}/issues?state=closed&sort=comments&direction=desc&per_page={count}"
    issues = fetch_data(url)
    return [issue for issue in issues if "pull_request" in issue and not issue.get('user', {}).get('type') == 'Bot']

def get_pr_details(repo, pr_number):
    """Fetches pull request details by PR number."""
    url = f"{GITHUB_API_URL}/repos/{repo}/pulls/{pr_number}"
    pr = fetch_data(url)
    return pr if pr else None

def get_pr_comments(repo, pr_number):
    """Fetches comments associated with a pull request."""
    url = f"{GITHUB_API_URL}/repos/{repo}/issues/{pr_number}/comments"
    comments = fetch_data(url)
    return [c for c in comments if not c.get('user', {}).get('type') == 'Bot']

def save_to_csv(repo, filename, fieldnames, data):
    """Saves the extracted data into CSV files with structured format."""
    os.makedirs(f"{CONFIG['github']['data_dir']}/{repo}", exist_ok=True)
    filepath = f"{CONFIG['github']['data_dir']}/{repo}/{filename}"
    with open(filepath, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)

def collect_repo_data(repo):
    """Collects PR data and comments for a given repository."""
    print(f"Collecting data for {repo}...")
    issues = get_most_commented_issues(repo)
    
    pr_data, comment_data = [], []
    for issue in issues:
        pr_number = issue['number']
        pr = get_pr_details(repo, pr_number)
        if pr:
            pr_data.append({
                'id': pr.get('id'), 'number': pr.get('number'), 'title': pr.get('title', 'N/A'),
                'user': pr.get('user', {}).get('login', 'unknown'), 'created_at': pr.get('created_at', 'N/A'),
                'merged_at': pr.get('merged_at', 'N/A'), 'comments': issue.get('comments', 0), 'state': pr.get('state', 'unknown')
            })
            
            comments = get_pr_comments(repo, pr_number)
            for comment in comments:
                comment_data.append({
                    'id': comment['id'], 'pr_number': pr_number, 'user': comment['user']['login'],
                    'created_at': comment['created_at'], 'body': comment['body']
                })
    
    save_to_csv(repo, CONFIG['github']['file_names']['pull_requests'], pr_data[0].keys(), pr_data)
    save_to_csv(repo, CONFIG['github']['file_names']['comments'], comment_data[0].keys(), comment_data) if comment_data else None
    
    print(f"Data collection completed for {repo}.")

def main():
    """Main function to iterate over repositories and collect data."""
    for repo in CONFIG['github']['repositories']:
        collect_repo_data(repo)
        time.sleep(CONFIG['github']['sleep_time'])  # To avoid hitting API rate limits

if __name__ == "__main__":
    main()