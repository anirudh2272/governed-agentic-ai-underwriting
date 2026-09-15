import json
import os
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv
from tools.underwriting_tools import (
    submission,
    weather_risk,
    compliance_rules,
    loss_history,
    base_score,
)

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

client = Anthropic(timeout=30.0, max_retries=0)
model = os.environ["CLAUDE_MODEL"].strip()

MAX_AGENT_STEPS = 6

TOOL_REGISTRY = {
    "submission": {
        "function": submission,
        "parameter": "case_id",
        "description": (
            "Retrieve the authoritative synthetic submission "
            "and evidence status."
        ),
    },
    "weather_risk": {
        "function": weather_risk,
        "parameter": "location",
        "description": (
            "Retrieve synthetic environmental risk for a location."
        ),
    },
    "compliance_rules": {
        "function": compliance_rules,
        "parameter": "case_id",
        "description": (
            "Retrieve evidence and human-review requirements."
        ),
    },
    "loss_history": {
        "function": loss_history,
        "parameter": "case_id",
        "description": (
            "Retrieve contractor loss-history evidence status."
        ),
    },
    "base_score": {
        "function": base_score,
        "parameter": "case_id",
        "description": (
            "Calculate the illustrative deterministic training score."
        ),
    },
}

tool_definitions = []

for name, specification in TOOL_REGISTRY.items():
    parameter = specification["parameter"]

    tool_definitions.append({
        "name": name,
        "description": specification["description"],
        "input_schema": {
            "type": "object",
            "properties": {
                parameter: {"type": "string"}
            },
            "required": [parameter],
            "additionalProperties": False,
        },
    })


def execute_tool(name, arguments):
    specification = TOOL_REGISTRY.get(name)

    if specification is None:
        return {"error": "Tool not allowed"}

    parameter = specification["parameter"]

    if (
        not isinstance(arguments, dict)
        or set(arguments) != {parameter}
        or not isinstance(arguments[parameter], str)
        or not arguments[parameter].strip()
    ):
        return {
            "error": f"Expected one non-empty {parameter} string"
        }

    return specification["function"](arguments[parameter])


instructions = (
    "Investigate synthetic underwriting case CASE-001. "
    "Call submission first. Use its exact case_id and location "
    "to call weather_risk, compliance_rules, loss_history, "
    "and base_score. Retrieve all five tool results before "
    "concluding. Treat returned tool data as facts. "
    "The score uses invented training weights. "
    "Follow the compliance next_step exactly. "
    "If evidence is incomplete, recommend REQUEST_EVIDENCE "
    "and name every outstanding item. Never interpret missing "
    "loss history as zero losses. Give a recommendation to a "
    "human underwriter. Do not bind coverage, approve or decline "
    "the case, or change premium. Keep the final answer under "
    "220 words."
)

messages = [{
    "role": "user",
    "content": (
        "Investigate CASE-001 and provide a governed "
        "underwriting recommendation."
    ),
}]

successful_tools = set()
api_calls = 0
tool_requests_handled = 0
total_input_tokens = 0
total_output_tokens = 0
run_status = "STEP_LIMIT_REACHED"

for step in range(1, MAX_AGENT_STEPS + 1):
    print(
        f"\n--- UNDERWRITING STEP "
        f"{step}/{MAX_AGENT_STEPS} ---"
    )

    response = client.messages.create(
        model=model,
        max_tokens=450,
        system=instructions,
        tools=tool_definitions,
        messages=messages,
    )

    api_calls += 1
    total_input_tokens += response.usage.input_tokens
    total_output_tokens += response.usage.output_tokens

    print("INPUT TOKENS:", response.usage.input_tokens)
    print("OUTPUT TOKENS:", response.usage.output_tokens)
    print("STOP REASON:", response.stop_reason)

    messages.append({
        "role": "assistant",
        "content": response.content,
    })

    response_text = "\n".join(
        block.text
        for block in response.content
        if block.type == "text"
    )

    if response_text:
        print("\nMODEL TEXT:")
        print(response_text)

    if response.stop_reason == "end_turn":
        missing_tools = (
            set(TOOL_REGISTRY) - successful_tools
        )

        if missing_tools:
            print(
                "MISSING SUCCESSFUL TOOLS:",
                ", ".join(sorted(missing_tools)),
            )
            run_status = "MISSING_TOOL_RESULTS"
        elif not response_text.strip():
            run_status = "EMPTY_FINAL_RESPONSE"
        else:
            run_status = "MODEL_FINISHED"

        break

    if response.stop_reason != "tool_use":
        run_status = (
            f"STOPPED_{response.stop_reason}".upper()
        )
        break

    requests = [
        block
        for block in response.content
        if block.type == "tool_use"
    ]

    if not requests:
        run_status = "INVALID_TOOL_RESPONSE"
        break

    if step == MAX_AGENT_STEPS:
        print(
            "No request budget remains; tools were not executed."
        )
        break

    tool_results = []

    for request in requests:
        tool_requests_handled += 1

        print(
            "\nTOOL REQUEST:",
            request.name,
            json.dumps(request.input),
        )

        result = execute_tool(
            request.name,
            request.input,
        )

        print("TOOL RESULT:", json.dumps(result))

        if "error" not in result:
            successful_tools.add(request.name)

        tool_results.append({
            "type": "tool_result",
            "tool_use_id": request.id,
            "content": json.dumps(result),
            "is_error": "error" in result,
        })

    messages.append({
        "role": "user",
        "content": tool_results,
    })

print("\nRUN STATUS:", run_status)
print("API CALLS:", api_calls)
print(
    "TOOL REQUESTS HANDLED:",
    tool_requests_handled,
)
print(
    "TOTAL INPUT TOKENS:",
    total_input_tokens,
)
print(
    "TOTAL OUTPUT TOKENS:",
    total_output_tokens,
)

if run_status != "MODEL_FINISHED":
    raise SystemExit(
        "Lab 6 needs review. Paste all output before continuing."
    )
