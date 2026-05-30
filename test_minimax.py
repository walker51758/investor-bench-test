#!/usr/bin/env python3
"""Test MiniMax API connectivity using OpenAI SDK"""

import os
import sys

# Set environment variables
os.environ["OPENAI_BASE_URL"] = "https://api.minimaxi.com/v1"
os.environ["OPENAI_API_KEY"] = "sk-cp--5NTgOHaAUI3aC3bsdFyyqgmW6qjMlkjTz5vSZjupOd0XACPvEHmNP7BpiINaNc2HCXw22FuGM_4P581n4rc5lns6M-PCn5ZV29jmdyGYda0a1QutjW8NKo"

print("Environment variables set:")
print(f"  OPENAI_BASE_URL: {os.environ['OPENAI_BASE_URL']}")
print(f"  OPENAI_API_KEY: {os.environ['OPENAI_API_KEY'][:20]}...")
print()

try:
    from openai import OpenAI
    client = OpenAI()

    print("Calling MiniMax Chat API...")
    response = client.chat.completions.create(
        model="MiniMax-M2.7",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say hello in one word"},
        ],
        max_tokens=20,
        temperature=1.0,
    )
    print(f"Success! Response: {response.choices[0].message.content}")

except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
    sys.exit(1)