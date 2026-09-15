import json
import os
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv
from tools.banking_tools import (
    get_customer,
    get_transaction,
    get_device_status,
    get_risk_rules,
)

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
client = Anthropic(timeout=30.0, max_retries=0)
model = os.environ["CLAUDE_MODEL"].strip()
MAX_AGENT_STEPS = 6

# This registry is the application's tool allowlist.
TOOL_REGISTRY = {
    "get_transaction": {
        "function": get_transaction,
        "parameter": "transaction_id",
        "description": "Get a fictional transaction's amount, currency, customer ID, and device ID.",
    },
    "get_customer": {
        "function": get_customer,
        "parameter": "customer_id",
        "description": "Get a fictional customer's risk label, typical amount, and device alert.",
    },
    "get_device_status": {
        "function": get_device_status,
        "parameter": "device_id",
        "description": "Get a fictional device's customer ID, new-device flag, and verification status.",
    },
    "get_risk_rules": {
        "function": get_risk_rules,
        "parameter": None,
        "description": "Get the fictional review rules, rule IDs, and how to combine them. Takes no arguments.",
    },
}

# Build the tool descriptions sent to Claude.
tool_definitions = []
for name, spec in TOOL_REGISTRY.items():
    parameter = spec["parameter"]
    properties = {parameter: {"type": "string"}} if parameter else {}
    tool_definitions.append({
        "name": name,
        "description": spec["description"],
        "input_schema": {
            "type": "object",
            "properties": properties,
            "required": list(properties),
            "additionalProperties": False,
        },
    })


def execute_tool(name, arguments):
    spec = TOOL_REGISTRY.get(name)
    if spec is None:
        return {"error": "Tool not allowed"}

    expected = {spec["parameter"]} if spec["parameter"] else set()
    if not isinstance(arguments, dict) or set(arguments) != expected:
        return {"error": f"Expected argument names: {sorted(expected)}"}

    if any(
        not isinstance(value, str) or not value.strip()
        for value in arguments.values()
    ):
        return {"error": "Argument values must be non-empty strings"}

    return spec["function"](**arguments)


instructions = (
    "Investigate the fictional transaction using get_transaction first. "
    "Use the returned customer and device IDs for their lookups. "
    "Retrieve the customer, device, and risk rules before concluding. "
    "Apply only the returned fictional rules. typical_max describes "
    "usual behavior, not an enforced transaction limit. "
    "Show your arithmetic and identify matched rule IDs. "
    "Separate facts from recommendations. Do not claim any transaction "
    "action was performed. Keep the final answer under 180 words."
)
messages = [{"role": "user", "content": "Investigate transaction T1001."}]

api_calls = 0
tool_requests_handled = 0
total_input = 0
total_output = 0
successful_tools = set()
run_status = "STEP_LIMIT_REACHED"

for step in range(1, MAX_AGENT_STEPS + 1):
    print(f"\n--- AGENT STEP {step}/{MAX_AGENT_STEPS} ---")
    api_calls += 1
    response = client.messages.create(
        model=model,
        max_tokens=400,
        system=instructions,
        tools=tool_definitions,
        messages=messages,
    )

    total_input += response.usage.input_tokens
    total_output += response.usage.output_tokens
    print("INPUT TOKENS:", response.usage.input_tokens)
    print("OUTPUT TOKENS:", response.usage.output_tokens)
    print("STOP REASON:", response.stop_reason)
    messages.append({"role": "assistant", "content": response.content})

    text = "\n".join(
        block.text for block in response.content if block.type == "text"
    )
    if text:
        print("\nMODEL TEXT:")
        print(text)

    if response.stop_reason == "end_turn":
        missing = set(TOOL_REGISTRY) - successful_tools
        if missing:
            run_status = "MISSING_TOOL_RESULTS"
            print("Missing successful lookups:", ", ".join(sorted(missing)))
        elif not text.strip():
            run_status = "EMPTY_RESPONSE"
        else:
            run_status = "MODEL_FINISHED"
        break

    if response.stop_reason != "tool_use":
        run_status = f"STOPPED_{response.stop_reason}".upper()
        break

    requests = [
        block for block in response.content if block.type == "tool_use"
    ]
    if not requests:
        run_status = "INVALID_TOOL_RESPONSE"
        break

    if step == MAX_AGENT_STEPS:
        print("No request budget remains; further tools will not execute.")
        break

    results = []
    for request in requests:
        tool_requests_handled += 1
        print("\nTOOL REQUEST:", request.name, json.dumps(request.input))
        result = execute_tool(request.name, request.input)
        print("TOOL RESULT:", json.dumps(result))

        if "error" not in result:
            successful_tools.add(request.name)

        results.append({
            "type": "tool_result",
            "tool_use_id": request.id,
            "content": json.dumps(result),
            "is_error": "error" in result,
        })

    messages.append({"role": "user", "content": results})

print("\nRUN STATUS:", run_status)
print("API CALLS:", api_calls)
print("TOOL REQUESTS HANDLED:", tool_requests_handled)
print("TOTAL INPUT TOKENS:", total_input)
print("TOTAL OUTPUT TOKENS:", total_output)

if run_status != "MODEL_FINISHED":
    raise SystemExit("Lab 4 needs review. Paste all output before continuing.")
