import os
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(env_path)


api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
model = os.getenv("CLAUDE_MODEL", "").strip()

print("API Key Loaded:", bool(api_key))
print("Model Configured:", bool(model))

if not api_key:
    raise SystemExit("ANTHROPIC_API_KEY is missing or blank.")

if not model:
    raise SystemExit("CLAUDE_MODEL is missing or blank.")

client = Anthropic(api_key=api_key)
print("Anthropic SDK initialized successfully.")
