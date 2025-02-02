import os

CONFIG = {
    "github": {
        "token": os.getenv("GITHUB_TOKEN"),  # Personal access token
        "base_url": "https://api.github.com",
        "rate_limit": {
            "max_requests": 5000,  # GitHub API rate limit per hour
            "min_remaining": 100   # Stop execution when remaining calls reach this
        },
        "repositories": [
            "kubernetes/kubernetes",
            # "redis/redis",
            # "apache/kafka",
            # "elementary/terminal",
            # "audacity/audacity",
            # "deluge-torrent/deluge",
            # "buildaworldnet/IrrlichtBAW",
            # "linuxmint/cinnamon-desktop",
            # "qBittorrent/qBittorrent"
        ],
        "pull_requests": {
            "top_n": 100  # Number of most commented PRs to retrieve
        },
        "file_names": {
            "pull_requests": "pull_requests.csv",
            "comments": "comments.csv",
            "reviews": "reviews.csv"
        },
        "data_dir": "data",  # Directory to store collected data
        "sleep_time": 2  # Time to sleep between API calls to avoid hitting rate limits
    }
}
