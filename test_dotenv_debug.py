#!/usr/bin/env python3
"""Debug dotenv loading issue"""

import os
from dotenv import load_dotenv

# Read .env file directly first
print("=== Raw .env file contents ===")
with open(".env", "rb") as f:
    raw = f.read()
print(f"File size: {len(raw)} bytes")
print()

# Show hex around MINIMAX_API_KEY
idx = raw.find(b"MINIMAX_API_KEY")
if idx >= 0:
    # Show from = to end of line
    start = idx + len(b"MINIMAX_API_KEY=")
    end = raw.find(b"\n", start)
    if end < 0:
        end = len(raw)
    key_bytes = raw[start:end]
    print(f"Raw key bytes ({len(key_bytes)} bytes): {key_bytes}")
    print(f"Key as string: {key_bytes.decode('utf-8')}")
    print(f"Key string length: {len(key_bytes.decode('utf-8'))}")
    # check for special chars
    for i, b in enumerate(key_bytes):
        if b < 32 or b > 126:
            print(f"  Special byte at position {i}: 0x{b:02x}")
        if b == ord(b'$'):
            print(f"  DOLLAR SIGN at position {i}")
else:
    print("MINIMAX_API_KEY not found in raw file!")

print()

# Compare with MINIMAX_EMB_API_KEY
idx2 = raw.find(b"MINIMAX_EMB_API_KEY")
if idx2 >= 0:
    start2 = idx2 + len(b"MINIMAX_EMB_API_KEY=")
    end2 = raw.find(b"\n", start2)
    if end2 < 0:
        end2 = len(raw)
    key2_bytes = raw[start2:end2]
    print(f"EMB key bytes ({len(key2_bytes)} bytes)")
    print(f"EMB key string length: {len(key2_bytes.decode('utf-8'))}")

print()

# Now check what dotenv loads
print("=== After load_dotenv() ===")
load_dotenv()
key = os.environ.get("MINIMAX_API_KEY", "NOT FOUND")
print(f"MINIMAX_API_KEY from environ: '{key[:30]}...' (total len: {len(key)})")
emb_key = os.environ.get("MINIMAX_EMB_API_KEY", "NOT FOUND")
print(f"MINIMAX_EMB_API_KEY from environ: '{emb_key[:30]}...' (total len: {len(emb_key)})")

# Also check if there's an env var already set that overrides it
print()
print("=== Check if env var was pre-set ===")
import subprocess
result = subprocess.run(["echo", "$MINIMAX_API_KEY"], shell=True, capture_output=True, text=True)
print(f"Shell echo: '{result.stdout.strip()}' (len: {len(result.stdout.strip())})")
