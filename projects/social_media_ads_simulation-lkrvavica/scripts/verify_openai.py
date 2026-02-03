
import os
import requests
import json
from dotenv import load_dotenv

load_dotenv()

def verify():
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        print("No Key")
        return

    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 5
    }
    
    print(f"Sending request to {url}...")
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=10)
        print(f"Status: {r.status_code}")
        print(f"Response: {r.text[:500]}")
    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    verify()
