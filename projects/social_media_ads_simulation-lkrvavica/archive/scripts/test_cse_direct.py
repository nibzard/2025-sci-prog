import os
import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv('GOOGLE_API_KEY')
cse_id = os.getenv('GOOGLE_CSE_ID')

print(f"API Key: {api_key[:20]}... (length: {len(api_key)})")
print(f"CSE ID: {cse_id}")

url = "https://www.googleapis.com/customsearch/v1"
params = {
    "key": api_key,
    "cx": cse_id,
    "q": "writer",
    "num": 3
}

print(f"\nTesting API call...")
try:
    response = requests.get(url, params=params)
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"Success! Found {len(data.get('items', []))} results")
        for item in data.get('items', [])[:2]:
            print(f"  - {item.get('link')}")
    else:
        print(f"Error Response:")
        print(response.text[:500])
        
except Exception as e:
    print(f"Exception: {e}")
