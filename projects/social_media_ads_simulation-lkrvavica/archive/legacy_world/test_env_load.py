import os
import sys
from dotenv import load_dotenv, find_dotenv

# Mimic the path setup in simulator.py
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

# Try to load
path = find_dotenv()
print(f"Found .env at: {path}")
load_dotenv(path)

key = os.getenv("OPENAI_API_KEY")
if key:
    print(f"Key loaded: {key[:5]}...")
else:
    print("Key NOT loaded")
