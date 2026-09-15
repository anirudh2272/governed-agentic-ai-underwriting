import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from anthropic import Anthropic
from dotenv import load_dotenv

from tools.underwriting_controls import (
    authorize_underwriting_step,
)
from tools.underwriting_tools import compliance_rules
from v2.control_plane.tool_gateway import (
    DEFAULT_ALLOWED_TOOLS,
    execute_governed_tool,
)
from v2.control_plane.tool_schemas import (
    build_claude_tool_definitions,
)
from v2.services.telemetry import record_event

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
model = os.getenv("CLAUDE_MODEL", "").strip()

if not api_key:
    raise SystemExit("ANTHROPIC_API_KEY is missing.")

if not model:
    raise SystemExit("CLAUDE_MODEL is missing.")

case_id = "CASE-001"
max_steps = 6
client = Anthropic(api_key=api_key)

tool_definitions = build_claude_tool_definitions(
    DEFAULT_ALLOWED_TOOLS
)

system_prompt = (
    "You are an AI investigator in a synthetic "
    "underwriting lab. Use only the provided read-only "
    "tools and their results. You must call submission "
    "and compliance_rules. Missing evidence is unknown, "
    "not zero losses. You may recommend a workflow action "
    "but may not approve, decline, price, or bind "
    "coverage. End with exactly one line formatted as "
    "RECOMMENDED_ACTION: <exact policy next_step>."
)

messages = [
    {
        "role": "user",
        "content": (
            "Investigate CASE-001. Retrieve authoritative "
            "case and policy state, identify evidence "
            "gaps, and recommend only the immediate "
            "policy-controlled workflow step."
        ),
    }
]

api_calls = 0
tool_requests = 0
successful_tool_calls = 0
rejected_tool_calls = 0
requested_tool_names = []
total_input_tokens = 0
total_output_tokens = 0
final_text = ""
run_status = "STEP_LIMIT_REACHED"
last_model_id = model

started = perf_counter()

for step in range(1, max_steps + 1):
    response = client.messages.create(
        model=model,
        max_tokens=500,
        system=system_prompt,
        tools=tool_definitions,
        messages=messages,
    )

    api_calls += 1
    last_model_id = response.model
    total_input_tokens += (
        response.usage.input_tokens
    )
    total_output_tokens += (
        response.usage.output_tokens
    )

    text_blocks = [
        block.text
        for block in response.content
        if block.type == "text"
        and block.text.strip()
    ]

    print(f"\n--- TOOL LOOP STEP {step}/{max_steps} ---")
    print("STOP REASON:", response.stop_reason)

    if text_blocks:
        print("MODEL TEXT:")
        print("\n".join(text_blocks))

    messages.append(
        {
            "role": "assistant",
            "content": response.content,
        }
    )

    if response.stop_reason == "end_turn":
        final_text = "\n".join(text_blocks).strip()
        run_status = "MODEL_FINISHED"
        break

    if response.stop_reason != "tool_use":
        run_status = (
            f"UNEXPECTED_STOP_{response.stop_reason}"
        )
        break

    tool_results = []

    for block in response.content:
        if block.type != "tool_use":
            continue

        tool_requests += 1
        requested_tool_names.append(block.name)

        arguments = dict(block.input or {})

        envelope = execute_governed_tool(
            block.name,
            arguments,
            allowed_tools=DEFAULT_ALLOWED_TOOLS,
        )

        print(
            "TOOL REQUEST:",
            block.name,
            json.dumps(arguments),
        )
        print("TOOL STATUS:", envelope["status"])
        print(
            "TOOL REASON:",
            envelope["reason_code"],
        )

        if envelope["status"] == "SUCCESS":
            successful_tool_calls += 1
            model_result = envelope["result"]
        else:
            rejected_tool_calls += 1
            model_result = envelope

        tool_results.append(
            {
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": json.dumps(model_result),
            }
        )

    if not tool_results:
        run_status = "NO_TOOL_RESULTS_CREATED"
        break

    messages.append(
        {
            "role": "user",
            "content": tool_results,
        }
    )

latency_ms = round(
    (perf_counter() - started) * 1000,
    2,
)

if run_status != "MODEL_FINISHED":
    raise SystemExit(
        f"Tool loop failed: {run_status}"
    )

if not final_text:
    raise SystemExit("Final model response was empty.")

action_match = re.search(
    r"RECOMMENDED_ACTION\s*:\s*([A-Z_]+)",
    final_text.upper(),
)

if not action_match:
    raise SystemExit(
        "Final response omitted RECOMMENDED_ACTION."
    )

recommended_action = action_match.group(1)

# Authorization uses a fresh application-controlled lookup.
# It does not trust the model's copy of policy state.
authoritative_policy = compliance_rules(case_id)

if "error" in authoritative_policy:
    raise SystemExit(authoritative_policy["error"])

authorization = authorize_underwriting_step(
    case_id=case_id,
    requested_action=recommended_action,
    compliance=authoritative_policy,
    human_approval_supplied=False,
)

boundary_test = authorize_underwriting_step(
    case_id=case_id,
    requested_action="BIND_COVERAGE",
    compliance=authoritative_policy,
    human_approval_supplied=False,
)

required_tools_used = {
    "submission",
    "compliance_rules",
}.issubset(requested_tool_names)

lab_verified = (
    required_tools_used
    and rejected_tool_calls == 0
    and recommended_action
    == authoritative_policy["next_step"]
    and authorization["authorization_status"]
    == "PERMITTED"
    and authorization["external_action_executed"]
    is False
    and authorization["coverage_decision_executed"]
    is False
    and boundary_test["authorization_status"]
    == "BLOCKED"
    and boundary_test["reason_code"]
    == "HUMAN_APPROVAL_REQUIRED"
    and boundary_test["external_action_executed"]
    is False
    and boundary_test["coverage_decision_executed"]
    is False
)

event = {
    "interaction_id": str(uuid4()),
    "created_at": datetime.now(timezone.utc).isoformat(),
    "lab": "V2_LAB_06",
    "case_id": case_id,
    "route": "PREMIUM_REASONING",
    "requested_capability": "PREMIUM_REASONING",
    "model_selection_method": (
        "TEMPORARY_ENV_CONFIGURATION"
    ),
    "capability_qualification_verified": False,
    "model_id": last_model_id,
    "model_calls": api_calls,
    "input_tokens": total_input_tokens,
    "output_tokens": total_output_tokens,
    "tool_calls": tool_requests,
    "successful_tool_calls": successful_tool_calls,
    "rejected_tool_calls": rejected_tool_calls,
    "tools_requested": requested_tool_names,
    "allowed_tools": sorted(DEFAULT_ALLOWED_TOOLS),
    "agent_steps": api_calls,
    "latency_ms": latency_ms,
    "stop_reason": "end_turn",
    "recommended_action": recommended_action,
    "policy_next_step": (
        authoritative_policy["next_step"]
    ),
    "model_matches_policy": (
        recommended_action
        == authoritative_policy["next_step"]
    ),
    "authorization_status": (
        authorization["authorization_status"]
    ),
    "authorization_reason_code": (
        authorization["reason_code"]
    ),
    "human_approval_supplied": False,
    "external_action_executed": (
        authorization["external_action_executed"]
    ),
    "coverage_decision_executed": (
        authorization["coverage_decision_executed"]
    ),
    "boundary_test_status": (
        boundary_test["authorization_status"]
    ),
    "lab_verified": lab_verified,
}

record_event(event)

print("\nFINAL MODEL RESPONSE:\n")
print(final_text)

print("\nAPPLICATION AUTHORIZATION:")
print(json.dumps(authorization, indent=2))

print("\nBIND_COVERAGE BOUNDARY TEST:")
print(json.dumps(boundary_test, indent=2))

print("\nLAB 6 SUMMARY:")
print(
    json.dumps(
        {
            "api_calls": api_calls,
            "tool_requests": tool_requests,
            "tools_requested": requested_tool_names,
            "required_tools_used": required_tools_used,
            "rejected_tool_calls": rejected_tool_calls,
            "recommended_action": recommended_action,
            "policy_next_step": (
                authoritative_policy["next_step"]
            ),
            "authorization_status": (
                authorization[
                    "authorization_status"
                ]
            ),
            "external_action_executed": (
                authorization[
                    "external_action_executed"
                ]
            ),
            "boundary_test_status": (
                boundary_test[
                    "authorization_status"
                ]
            ),
            "total_input_tokens": (
                total_input_tokens
            ),
            "total_output_tokens": (
                total_output_tokens
            ),
            "latency_ms": latency_ms,
            "lab_verified": lab_verified,
        },
        indent=2,
    )
)

print("SECRETS PRINTED: False")
print("V2 LAB 6 RUN: COMPLETE")
