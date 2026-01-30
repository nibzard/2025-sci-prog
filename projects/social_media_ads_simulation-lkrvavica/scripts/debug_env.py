
import os
from dotenv import load_dotenv

load_dotenv()

key = os.getenv("OPENAI_API_KEY")
google_key = os.getenv("GOOGLE_API_KEY")

print(f"OPENAI_API_KEY Present: {bool(key)}")
print(f"GOOGLE_API_KEY Present: {bool(google_key)}")
if google_key:
    print(f"Google Key Start: {google_key[:5]}...")
if key:
    print(f"Key Length: {len(key)}")
    print(f"Key Start: {key[:5]}...")
else:
    print("OPENAI_API_KEY is None or Empty")
