import os
import requests
import json
from urllib.parse import urlencode
from dotenv import load_dotenv

load_dotenv()

SERP_PROXY_USERNAME = os.environ.get("BRIGHTDATA_SERP_USERNAME")
SERP_PROXY_PASSWORD = os.environ.get("BRIGHTDATA_SERP_PASSWORD")

def search_clips(query):
    if not SERP_PROXY_USERNAME:
        print("No creds")
        return []
        
    proxies = {
        "http":  f"http://{SERP_PROXY_USERNAME}:{SERP_PROXY_PASSWORD}@brd.superproxy.io:33335",
        "https": f"http://{SERP_PROXY_USERNAME}:{SERP_PROXY_PASSWORD}@brd.superproxy.io:33335",
    }
    
    params = {
        "q": f'site:twitch.tv/clip "{query}"',
        "num": 10,
        "hl": "en", 
        "gl": "us",
    }
    
    url = "https://www.google.com/search?" + urlencode(params)
    headers = {"brd-json": "1"}
    
    try:
        print(f"Searching for {query}...")
        resp = requests.get(url, headers=headers, proxies=proxies, verify=False, timeout=30)
        data = resp.json()
        
        urls = []
        for item in data.get("organic", []):
            l = item.get("link")
            if l and "/clip/" in l:
                urls.append(l)
        return urls
    except Exception as e:
        print(e)
        return []

categories = ["gaming", "chatting", "coding", "music", "sports"]
results = {}

for c in categories:
    urls = search_clips(c)
    results[c] = urls
    print(f"{c}: Found {len(urls)}")

print("\n--- RESULTS ---")
print(json.dumps(results, indent=2))
