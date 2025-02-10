import os
import requests
import datetime

HEADERS = {
    "Authorization": f"token {os.getenv("GITHUB_TOKEN")}",
    "Accept": "application/vnd.github.v3+json"
}

def check_api_rate_limit():
    url = "https://api.github.com/rate_limit"
    response = requests.get(url, headers=HEADERS)
    data = response.json()
    rate = data.get("rate", {})
    remaining = rate.get("remaining")
    limit = rate.get("limit")
    reset_timestamp = rate.get("reset")
    reset_time = datetime.datetime.fromtimestamp(reset_timestamp)
    
    print(f"API calls remaining: {remaining} out of {limit}")
    print(f"Rate limit resets at: {reset_time}")

if __name__ == "__main__":
    check_api_rate_limit()