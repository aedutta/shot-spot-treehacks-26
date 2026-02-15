import os
import requests
import json
from urllib.parse import urlencode
from dotenv import load_dotenv

# Load env vars from the root .env file
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv(os.path.join(project_root, ".env"))

print("🔎 Testing Bright Data Direct API (SERP)...")

# 1. Credentials
api_token = os.environ.get("BRIGHTDATA_API_KEY")
zone_name = "serp_api1"

if not api_token:
    print("❌ Error: BRIGHTDATA_API_KEY not found in .env")
    exit(1)

print(f"✅ API Key found: {api_token[:5]}...")
print(f"   Target Zone: {zone_name}")

# 2. Construct Request
url = "https://api.brightdata.com/request"
headers = {
    "Authorization": f"Bearer {api_token}",
    "Content-Type": "application/json",
}

query = "site:youtube.com funny cat videos"
# Properly encode parameters - REMOVED 'num' as warned by API
params = {"q": query, "hl": "en", "gl": "us"}
# Appending brd_json=1 to force SERP parsing
google_search_url = "https://www.google.com/search?" + urlencode(params) + "&brd_json=1"

payload = {
    "zone": zone_name,
    "url": google_search_url,
    "format": "json"
}

print(f"\n🚀 Sending request for: '{query}'")
print(f"   Endpoint: {url}")
print(f"   Target URL: {google_search_url}")

try:
    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    
    print(f"   Status Code: {resp.status_code}")
    
    if resp.status_code == 200:
        try:
            data = resp.json()
            
            # Check if we have a wrapped response (status_code + body) or direct results
            # If brd_json works, the 'body' might be the parsed JSON string, OR the top level object is the result.
            
            organic = []
            
            if "organic" in data:
                # Direct result
                organic = data["organic"]
            elif "body" in data:
                # Wrapped result - check if body is JSON string
                print("   Response is wrapped (has 'body' field).")
                body_content = data["body"]
                try:
                    if isinstance(body_content, str):
                        print("   Body is string, attempting parse...")
                        body_json = json.loads(body_content)
                        organic = body_json.get("organic", [])
                    elif isinstance(body_content, dict):
                         print("   Body is dict...")
                         organic = body_content.get("organic", [])
                except Exception as e:
                     print(f"   ⚠️ Could not parse body as JSON: {e}")
                     print(f"   Body start: {str(body_content)[:100]}")

            if len(organic) == 0:
                print("⚠️ 0 Organic results found. Dumping response structure:")
                # Print available top-level keys
                print(f"   Keys available: {list(data.keys())}")
                if "body" in data:
                     print(f"   Body Type: {type(data['body'])}")
                     print(f"   Body snippet: {str(data['body'])[:500]}")
            else:
                print(f"✅ Success! Found {len(organic)} organic results.")
                
                for i, item in enumerate(organic[:3]):
                    link = item.get("link") or item.get("url")
                    title = item.get("title")
                    print(f"   Result {i+1}: {title} ({link})")
                
        except Exception as e:
            print(f"⚠️ Response was 200 OK but JSON parsing failed: {e}")
            print(f"   Raw Response: {resp.text[:500]}...")
            
    elif resp.status_code == 401:
        print("❌ 401 Unauthorized. Your API Key might be invalid.")
    elif resp.status_code == 403:
        print("❌ 403 Forbidden. Check if your Zone 'serp_api1' is active and you have permissions.")
    else:
        print(f"❌ API Error: {resp.text[:500]}")

except Exception as e:
    print(f"❌ Connection Failed: {e}")
