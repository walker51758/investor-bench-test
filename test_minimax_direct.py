#!/usr/bin/env python3
"""Test MiniMax API using the EXACT same method as the project's openai_compatible.py"""

import os
import json
import httpx
from dotenv import load_dotenv

load_dotenv()

# === Test 1: Exact project request ===
print("=" * 60)
print("Test 1: Exact project request (httpx)")
print("=" * 60)

api_key = os.environ.get("MINIMAX_API_KEY")
print(f"API Key found: {'Yes' if api_key else 'No'}")
print(f"API Key prefix: {api_key[:15] if api_key else 'N/A'}...")
print(f"API Key length: {len(api_key) if api_key else 0}")

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
}

request_data = {
    "model": "MiniMax-M2.7",
    "max_tokens": 4096,
    "messages": [
        {"content": "You are a helpful assistant.", "role": "system"},
        {"content": "Say hello in one word", "role": "user"},
    ],
    "temperature": 1.0,
}

url = "https://api.minimaxi.com/v1/chat/completions"

try:
    with httpx.Client(timeout=30) as client:
        response = client.post(url=url, headers=headers, json=request_data)
    print(f"Status code: {response.status_code}")
    if response.status_code == 200:
        resp_json = response.json()
        content = resp_json["choices"][0]["message"]["content"]
        print(f"Success! Response: {content}")
    else:
        print(f"Error response: {response.text}")
except Exception as e:
    print(f"Exception: {type(e).__name__}: {e}")

# === Test 2: Check if Bearer prefix matters ===
print()
print("=" * 60)
print("Test 2: NO Bearer prefix")
print("=" * 60)

headers2 = {
    "Authorization": api_key,
    "Content-Type": "application/json",
}

try:
    with httpx.Client(timeout=30) as client:
        response = client.post(url=url, headers=headers2, json=request_data)
    print(f"Status code: {response.status_code}")
    if response.status_code == 200:
        print("Success!")
    else:
        print(f"Error: {response.text}")
except Exception as e:
    print(f"Exception: {type(e).__name__}: {e}")

# === Test 3: Check if trailing spaces in .env key ===
print()
print("=" * 60)
print("Test 3: Key read directly (bypass dotenv)")
print("=" * 60)

with open(".env", "r") as f:
    for line in f:
        line = line.strip()
        if line.startswith("MINIMAX_API_KEY="):
            raw_key = line.split("=", 1)[1]
            print(f"Raw key from file, first 15 chars: {raw_key[:15]}")
            print(f"Raw key length: {len(raw_key)}")
            print(f"Has trailing whitespace: {repr(raw_key[-5:])}")
            break
