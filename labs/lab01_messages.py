import os
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(env_path)

client = Anthropic(timeout=30.0, max_retries=0)
model = os.environ["CLAUDE_MODEL"].strip()

response = client.messages.create(
    model=model,
    max_tokens=200,
    messages=[
        {
            "role": "user",
            "content": "Explain enterprise architecture in three sentences."
        }
    ]
)

print("CLAUDE RESPONSE:")
for block in response.content:
    if block.type == "text":
        print(block.text)

print("\nINPUT TOKENS:", response.usage.input_tokens)
print("OUTPUT TOKENS:", response.usage.output_tokens)
print("STOP REASON:", response.stop_reason)
