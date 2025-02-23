import os

CONFIG = {
    "github": {
        "token": os.getenv("GITHUB_TOKEN"),  # Personal access token
        "base_url": "https://api.github.com",
        "repositories": [
            # "kubernetes/kubernetes", #1
            # "redis/redis", #2
            # "apache/kafka", #3
            # "elementary/terminal", #4
            # "audacity/audacity", #5
            # "deluge-torrent/deluge", #6
            # "buildaworldnet/IrrlichtBAW", #7
            # "linuxmint/cinnamon-desktop", #8
            "qBittorrent/qBittorrent", #9
            # "CivMC/Civ", #10
        ],
        "pull_requests": {
            "top_n": 100  # Number of PRs to retrieve
        },
        "file_names": {
            "pull_requests": "pull_requests.csv",
            "comments": "comments.csv",
            "reviews": "reviews.csv"
        },
        "raw_dir": "data/raw_v_02",
        # "raw_dir": "data/raw",

        "processed_dir": "data/processed",
    }
}
