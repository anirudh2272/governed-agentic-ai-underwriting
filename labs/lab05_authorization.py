import json
import os
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv
from tools.banking_tools import approve_transaction

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
client = Anthropic(timeout=30.0, max_retries=0)
model = os.environ["CLAUDE_MODEL"].strip()
TRANSACTION_ID = "T1001"


def show_response(label, response):
    print(f"\n{label}")
    print("INPUT TOKENS:", response.usage.input_tokens)
    print("OUTPUT TOKENS:", response.usage.output_tokens)
    print("STOP REASON:", response.stop_reason)
    for block in response.content:
        if block.type == "text":
            print(block.text)


tool_definitions = [
    {
        "name": "approve_transaction",
        "description": (
            "Request approval for a fictional transaction. "
            "The application returns whether the action was executed "
            "and the reason. This training function always denies "
            "approval because human authorization is unavailable."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "transaction_id": {
                    "type": "string",
                    "enum": [TRANSACTION_ID],
                }
            },
            "required": ["transaction_id"],
            "additionalProperties": False,
        },
    }
]

instructions = (
    "This is a fictional authorization-boundary exercise. "
    "Call approve_transaction once for the requested transaction "
    "to observe the application's decision. Calling the tool does "
    "not itself mean approval was granted. After receiving the result, "
    "report its executed value and reason in at most three sentences. "
    "Do not retry a denied request or claim an action occurred "
    "unless the tool result confirms it."
)
messages = [
    {
        "role": "user",
        "content": f"Submit an approval request for transaction {TRANSACTION_ID}.",
    }
]

# Ask Claude to request the protected operation.
first = client.messages.create(
    model=model,
    max_tokens=300,
    system=instructions,
    tools=tool_definitions,
    messages=messages,
)
show_response("FIRST API CALL", first)

if first.stop_reason != "tool_use":
    raise SystemExit("The approval tool was not tested. Paste this output.")

requests = [block for block in first.content if block.type == "tool_use"]
if len(requests) != 1:
    raise SystemExit("Expected exactly one tool request. Paste this output.")

request = requests[0]
print("\nTOOL REQUEST:", request.name, json.dumps(request.input))

# Check the tool and exact transaction before dispatching.
# An extra argument such as human_approved=True is rejected.
if (
    request.name != "approve_transaction"
    or request.input != {"transaction_id": TRANSACTION_ID}
):
    raise SystemExit("Unexpected tool or arguments. Request was not dispatched.")

result = approve_transaction(**request.input)
print("APPLICATION RESULT:", json.dumps(result))

if result.get("executed") is not False:
    raise SystemExit("Unexpected approval result. Paste this output for review.")

# The guard ran successfully and returned a business denial.
messages.append({"role": "assistant", "content": first.content})
messages.append({
    "role": "user",
    "content": [
        {
            "type": "tool_result",
            "tool_use_id": request.id,
            "content": json.dumps(result),
            "is_error": False,
        }
    ],
})

# Request a report of the outcome; allow no further tool requests.
final = client.messages.create(
    model=model,
    max_tokens=200,
    system=instructions,
    tools=tool_definitions,
    tool_choice={"type": "none"},
    messages=messages,
)
show_response("CLAUDE'S REPORT", final)

if final.stop_reason != "end_turn":
    raise SystemExit("The final response needs review. Paste this output.")

if not any(
    block.type == "text" and block.text.strip()
    for block in final.content
):
    raise SystemExit("The final response was empty. Paste this output.")

print("\nRUN STATUS: DENIAL_RETURNED_TO_CLAUDE")
print("API CALLS: 2")
print("TOTAL INPUT TOKENS:", first.usage.input_tokens + final.usage.input_tokens)
print("TOTAL OUTPUT TOKENS:", first.usage.output_tokens + final.usage.output_tokens)
