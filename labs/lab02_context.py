import os
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(env_path)

client = Anthropic(timeout=30.0, max_retries=0)
model = os.environ["CLAUDE_MODEL"].strip()


def show_result(label, response):
    print(f"\n{label}")

    for block in response.content:
        if block.type == "text":
            print(block.text)

    print("INPUT TOKENS:", response.usage.input_tokens)
    print("OUTPUT TOKENS:", response.usage.output_tokens)
    print("STOP REASON:", response.stop_reason)


customer_prompt = (
    "Customer C1001 has MEDIUM risk. "
    "Typical transaction maximum is $2,000. "
    "A new-device alert is active. "
    "Summarize this customer in two sentences."
)

question = "What customer were we discussing previously?"

# Call 1: Supply the customer facts.
first = client.messages.create(
    model=model,
    max_tokens=120,
    messages=[
        {"role": "user", "content": customer_prompt}
    ],
)
show_result("1. FIRST CALL — CUSTOMER FACTS", first)

# Call 2: Send only the follow-up question.
second = client.messages.create(
    model=model,
    max_tokens=120,
    messages=[
        {"role": "user", "content": question}
    ],
)
show_result("2. NEW CALL WITHOUT HISTORY", second)

# Call 3: Explicitly include the earlier conversation.
third = client.messages.create(
    model=model,
    max_tokens=120,
    messages=[
        {"role": "user", "content": customer_prompt},
        {"role": "assistant", "content": first.content},
        {"role": "user", "content": question},
    ],
)
show_result("3. CALL WITH HISTORY INCLUDED", third)
