import os
import requests
import csv
from src.config import CONFIG
from dateutil import parser

# GitHub API Headers for authentication and request handling
HEADERS = {
    "Authorization": f"token {CONFIG['github']['token']}",
    "Accept": "application/vnd.github.v3+json"
}

# Base URL for GitHub API requests
GITHUB_API_URL = CONFIG['github']['base_url']
# Number of pull requests to fetch (most commented)
COUNT = CONFIG['github']['pull_requests']['top_n']

def fetch_data(url):
    """Fetches data from the given GitHub API URL"""
    response = requests.get(url, headers=HEADERS)
    if response.status_code == 200:
        return response.json()
    print(f"Error fetching {url}: {response.status_code}")
    return []

def fetch_paginated_data(url):
    """Fetches all pages of data from a paginated GitHub API endpoint."""
    results = []
    while url:
        response = requests.get(url, headers=HEADERS)
        if response.status_code != 200:
            print(f"Error fetching {url}: {response.status_code}")
            break

        page_data = response.json()
        if isinstance(page_data, list):
            results.extend(page_data)
        else:
            # For non-list responses (like the search API), return immediately
            return page_data

        # Parse the Link header for next page URL if available
        link_header = response.headers.get('Link', None)
        next_url = None
        if link_header:
            parts = link_header.split(',')
            for part in parts:
                if 'rel="next"' in part:
                    next_url = part.split(';')[0].strip().strip('<>').strip()
                    break
        url = next_url
    return results

def get_most_commented_prs(repo, count=COUNT):
    """
    Uses the GitHub Search API to fetch the top commented closed pull requests.
    """
    print(f"Fetching most commented PRs for {repo}")
    search_url = (f"{GITHUB_API_URL}/search/issues"
                  f"?q=repo:{repo}+is:pr+is:closed"
                  f"&sort=comments&order=desc&per_page={count}")
    result = fetch_data(search_url)
    if result and "items" in result:
        return result["items"]
    return []

def get_pr_details(repo, pr_number):
    """Fetches pull request details by PR number."""
    print(f"Fetching details for PR #{pr_number} in {repo}")
    url = f"{GITHUB_API_URL}/repos/{repo}/pulls/{pr_number}"
    pr = fetch_data(url)
    return pr if pr else None

def get_pr_comments(repo, pr_number):
    """Fetches all issue comments associated with a pull request"""
    url = f"{GITHUB_API_URL}/repos/{repo}/issues/{pr_number}/comments?per_page=100"
    comments = fetch_paginated_data(url)
    return [c for c in comments if c.get('user', {}).get('type') != 'Bot']

def get_pr_reviews(repo, pr_number):
    """Fetches all reviews for a pull request"""
    url = f"{GITHUB_API_URL}/repos/{repo}/pulls/{pr_number}/reviews?per_page=100"
    reviews = fetch_paginated_data(url)
    if not reviews:
        reviews = []
    return [
        r for r in reviews 
        if r and r.get('user') and isinstance(r.get('user'), dict) and r.get('user').get('type') != 'Bot'
    ]

def get_pr_review_comments(repo, pr_number):
    """Fetches inline review comments for a PR."""
    url = f"{GITHUB_API_URL}/repos/{repo}/pulls/{pr_number}/comments?per_page=100"
    review_comments = fetch_paginated_data(url)

    if not isinstance(review_comments, list):
        print(f"Unexpected data format for PR#{pr_number} review comments: {review_comments}")
        return []

    # Extract required fields safely with proper error handling
    filtered_comments = []
    for rc in review_comments:
        if not isinstance(rc, dict):
            continue
            
        # Safely extract user information
        user = rc.get('user', {})
        if not isinstance(user, dict):
            continue
            
        login = user.get('login', 'unknown')
        
        filtered_comments.append({
            'id': rc.get('id', 'N/A'),
            'pr_number': pr_number,
            'user': login,
            'created_at': rc.get('created_at', 'N/A'),
            'body': rc.get('body', '')
        })
    
    return filtered_comments

def save_to_csv(repo, filename, fieldnames, data):
    """Saves the extracted data into CSV files with structured format."""
    print(f"Saving {filename} for {repo}")
    os.makedirs(f"{CONFIG['github']['raw_dir']}/{repo}", exist_ok=True)
    filepath = f"{CONFIG['github']['raw_dir']}/{repo}/{filename}"
    with open(filepath, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)

def save_summary_data(repo, comment_data, review_data, review_comments_data):
    """Saves summary data including total comments per PR."""
    summary_data = []
    pr_comment_counts = {}
    
    for comment in comment_data:
        pr_number = comment['pr_number']
        pr_comment_counts[pr_number] = pr_comment_counts.get(pr_number, 0) + 1
    
    for review in review_data:
        pr_number = review['pr_number']
        pr_comment_counts[pr_number] = pr_comment_counts.get(pr_number, 0) + 1
    
    for review_comment in review_comments_data:
        pr_number = review_comment['pr_number']
        pr_comment_counts[pr_number] = pr_comment_counts.get(pr_number, 0) + 1
    
    for pr_number, total_comments in pr_comment_counts.items():
        summary_data.append({'pr_number': pr_number, 'total_comments': total_comments})
    
    print(f"Total PRs with comments collected: {len(summary_data)}")
    save_to_csv(repo, "summary.csv", ['pr_number', 'total_comments'], summary_data)

def collect_repo_data(repo):
    """Collects PR data, comments, and reviews for a given repository."""
    print(f"Collecting data for {repo}...")
    pr_issues = get_most_commented_prs(repo)
    
    pr_data = []
    comment_data = []
    review_data = []
    review_comments_data = []
    
    for issue in pr_issues:
        pr_number = issue['number']
        pr = get_pr_details(repo, pr_number)
        if pr:
            pr_data.append({
                'id': pr.get('id'),
                'number': pr.get('number'),
                'title': pr.get('title', 'N/A'),
                'user': pr.get('user', {}).get('login', 'unknown'),
                'created_at': pr.get('created_at', 'N/A'),
                'merged_at': pr.get('merged_at', 'N/A'),
                'comments': issue.get('comments', 0),
                'state': pr.get('state', 'unknown'),
                'closed_at': pr.get('closed_at') 
            })

            # Collect regular comments
            comments = get_pr_comments(repo, pr_number)
            for comment in comments:
                comment_data.append({
                    'id': comment['id'],
                    'pr_number': pr_number,
                    'user': comment['user']['login'],
                    'created_at': comment['created_at'],
                    'body': comment['body']
                })

            # Collect review data
            reviews = get_pr_reviews(repo, pr_number)
            for review in reviews:
                if not review.get('submitted_at'):
                    continue

                review_data.append({
                    'id': review['id'],
                    'pr_number': pr_number,
                    'user': review['user']['login'],
                    'submitted_at': review.get('submitted_at', 'N/A'),
                    'state': review.get('state', 'N/A'),
                    'body': review.get('body', '')
                })

            # Collect review comments with the fixed implementation
            new_review_comments = get_pr_review_comments(repo, pr_number)
            review_comments_data.extend(new_review_comments)

    # Save collected data to CSV files
    if pr_data:
        save_to_csv(repo, CONFIG['github']['file_names']['pull_requests'], pr_data[0].keys(), pr_data)
    if comment_data:
        save_to_csv(repo, CONFIG['github']['file_names']['comments'], comment_data[0].keys(), comment_data)
    if review_data:
        save_to_csv(repo, CONFIG['github']['file_names']['reviews'], review_data[0].keys(), review_data)
    if review_comments_data:
        save_to_csv(repo, CONFIG['github']['file_names']['review_comments'], review_comments_data[0].keys(), review_comments_data)
    else:
        save_to_csv(repo, CONFIG['github']['file_names']['review_comments'], ['id', 'pr_number', 'user', 'created_at', 'body'], [])
    
    save_summary_data(repo, comment_data, review_data, review_comments_data)
    
    print(f"Data collection completed for {repo}.")

def main():
    """Main function to iterate over repositories and collect data."""
    for repo in CONFIG['github']['repositories']:
        collect_repo_data(repo)

if __name__ == "__main__":
    main()