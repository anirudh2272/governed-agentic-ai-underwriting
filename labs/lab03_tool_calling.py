import json
import os
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv

from tools.banking_tools import get_customer

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(env_path)

client = Anthropic(timeout=30.0, max_retries=0)
model = os.environ["CLAUDE_MODEL"].strip()


def show_usage(label, response):
    print(f"\n{label}")
    print("INPUT TOKENS:", response.usage.input_tokens)
    print("OUTPUT TOKENS:", response.usage.output_tokens)
    print("STOP REASON:", response.stop_reason)


# Describe the tool and the input it accepts.
tool_definitions = [
    {
        "name": "get_customer",
        "description": (
            "Look up a fictional customer by customer_id. "
            "Returns the risk label, typical transaction maximum, "
            "and new-device alert flag. The typical maximum "
            "describes usual behavior, not an approval limit."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "customer_id": {"type": "string"}
            },
            "required": ["customer_id"],
            "additionalProperties": False,
        },
    }
]

instructions = (
    "Analyze fictional training transactions. Retrieve the customer "
    "record with get_customer before giving an analysis. "
    "typical_max describes usual behavior, not an enforced limit. "
    "No approval or verification policy has been supplied. "
    "Separate retrieved facts from recommendations. "
    "Do not claim that any transaction action was performed. "
    "Keep your final answer under 120 words."
)

messages = [
    {
        "role": "user",
        "content": "Analyze a $25,000 transaction for customer C1001.",
    }
]

# First API call: Claude can request the customer lookup.
response = client.messages.create(
    model=model,
    max_tokens=300,
    system=instructions,
    tools=tool_definitions,
    messages=messages,
)
show_usage("FIRST API CALL", response)

if response.stop_reason != "tool_use":
    for block in response.content:
        if block.type == "text":
            print(block.text)
    raise SystemExit("Expected a tool request. Paste this output for review.")

# Preserve Claude's complete response in the conversation.
messages.append({
    "role": "assistant",
    "content": response.content,
})

tool_results = []

# Python checks each request and executes the allowed function.
for block in response.content:
    if block.type != "tool_use":
        continue

    print("\nCLAUDE REQUESTED TOOL:", block.name)
    print("CLAUDE SUPPLIED:", json.dumps(block.input))

    arguments = block.input

    if block.name != "get_customer":
        result = {"error": "Tool not allowed"}
    elif (
        not isinstance(arguments, dict)
        or set(arguments) != {"customer_id"}
        or not isinstance(arguments["customer_id"], str)
    ):
        result = {"error": "Expected one customer_id string"}
    else:
        result = get_customer(arguments["customer_id"])

    print("APPLICATION RESULT:", json.dumps(result))

    tool_results.append({
        "type": "tool_result",
        "tool_use_id": block.id,
        "content": json.dumps(result),
        "is_error": "error" in result,
    })

if not tool_results:
    raise SystemExit("No tool requests found. Paste this output for review.")

# Return the results immediately after Claude's tool request.
messages.append({
    "role": "user",
    "content": tool_results,
})

# Second API call: Claude receives the retrieved customer facts.
final = client.messages.create(
    model=model,
    max_tokens=300,
    system=instructions,
    tools=tool_definitions,
    messages=messages,
)
show_usage("SECOND API CALL", final)

print("\nCLAUDE ANALYSIS:")
for block in final.content:
    if block.type == "text":
        print(block.text)

if final.stop_reason != "end_turn":
    raise SystemExit("Response needs review. Paste the output before proceeding.")
