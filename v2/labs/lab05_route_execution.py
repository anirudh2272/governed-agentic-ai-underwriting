import json
import os
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from anthropic import Anthropic
from dotenv import load_dotenv

from v2.control_plane.router import route_request
from v2.services.context_builder import (
    build_explicit_context,
)
from v2.services.route_executor import execute_route
from v2.services.telemetry import record_event

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
configured_model = os.getenv(
    "CLAUDE_MODEL",
    "",
).strip()

if not api_key:
    raise SystemExit("ANTHROPIC_API_KEY is missing.")

if not configured_model:
    raise SystemExit("CLAUDE_MODEL is missing.")

client = Anthropic(api_key=api_key)
experiment_id = str(uuid4())

conversation_decision = route_request(
    task_type="EXPLAIN_EVIDENCE_STATUS",
    deterministic_answer=None,
    complexity="LOW",
    business_risk="LOW",
)

conversation_system = (
    "You are a concise workflow explainer in a synthetic "
    "underwriting lab. Use only supplied facts. Explain "
    "workflow status but do not make an underwriting "
    "decision or add requirements."
)

conversation_messages = [
    {
        "role": "user",
        "content": (
            "In exactly two sentences, explain that "
            "CASE-001 has evidence status INCOMPLETE "
            "because wind_mitigation and "
            "contractor_loss_history are outstanding, "
            "and that its policy next step is "
            "REQUEST_EVIDENCE."
        ),
    }
]

premium_decision = route_request(
    task_type="COMPLEX_RISK_ANALYSIS",
    deterministic_answer=None,
    complexity="HIGH",
    business_risk="HIGH",
)

premium_context = build_explicit_context("CASE-001")

conversation_result = execute_route(
    conversation_decision,
    client=client,
    model_id=configured_model,
    system=conversation_system,
    messages=conversation_messages,
    max_tokens=180,
)

premium_result = execute_route(
    premium_decision,
    client=client,
    model_id=configured_model,
    system=premium_context["system"],
    messages=premium_context["messages"],
    max_tokens=450,
)


def context_characters(
    system: str,
    messages: list[dict],
) -> int:
    return len(system) + sum(
        len(message["content"])
        for message in messages
        if isinstance(message.get("content"), str)
    )


def save_execution(
    *,
    task_type: str,
    result: dict,
    system: str,
    messages: list[dict],
    business_outcome: str,
) -> dict:
    event = {
        "interaction_id": str(uuid4()),
        "experiment_id": experiment_id,
        "created_at": (
            datetime.now(timezone.utc).isoformat()
        ),
        "lab": "V2_LAB_05",
        "case_id": "CASE-001",
        "task_type": task_type,
        "route": result["route"],
        "requested_capability": (
            result["requested_capability"]
        ),
        "execution_mode": (
            result["execution_mode"]
        ),
        "model_selection_method": (
            "TEMPORARY_ENV_CONFIGURATION"
        ),
        "capability_qualification_verified": False,
        "model_id": result["model_id"],
        "model_calls": result["model_calls"],
        "input_tokens": result["input_tokens"],
        "output_tokens": result["output_tokens"],
        "cache_creation_input_tokens": (
            result["cache_creation_input_tokens"]
        ),
        "cache_read_input_tokens": (
            result["cache_read_input_tokens"]
        ),
        "context_characters": context_characters(
            system,
            messages,
        ),
        "response_characters": len(
            result["response_text"]
        ),
        "tool_calls": 0,
        "agent_steps": 0,
        "latency_ms": result["latency_ms"],
        "stop_reason": result["stop_reason"],
        "business_outcome": business_outcome,
        "external_action_executed": False,
        "coverage_decision_executed": False,
    }

    record_event(event)
    return event


conversation_event = save_execution(
    task_type="EXPLAIN_EVIDENCE_STATUS",
    result=conversation_result,
    system=conversation_system,
    messages=conversation_messages,
    business_outcome="WORKFLOW_EXPLANATION_CREATED",
)

premium_event = save_execution(
    task_type="COMPLEX_RISK_ANALYSIS",
    result=premium_result,
    system=premium_context["system"],
    messages=premium_context["messages"],
    business_outcome="AI_RISK_RECOMMENDATION_CREATED",
)

execution_valid = (
    conversation_event["route"] == "CONVERSATIONAL"
    and premium_event["route"]
    == "PREMIUM_REASONING"
    and conversation_event["model_calls"] == 1
    and premium_event["model_calls"] == 1
    and conversation_event["input_tokens"] > 0
    and premium_event["input_tokens"] > 0
    and conversation_event[
        "external_action_executed"
    ]
    is False
    and premium_event[
        "coverage_decision_executed"
    ]
    is False
)

print("\nCONVERSATIONAL RESPONSE:\n")
print(conversation_result["response_text"])

print("\nPREMIUM-ROUTE RESPONSE:\n")
print(premium_result["response_text"])

print("\nROUTE EXECUTION TELEMETRY:")
print(
    json.dumps(
        {
            "conversational": {
                "route": conversation_event["route"],
                "requested_capability": (
                    conversation_event[
                        "requested_capability"
                    ]
                ),
                "model_calls": (
                    conversation_event["model_calls"]
                ),
                "input_tokens": (
                    conversation_event["input_tokens"]
                ),
                "output_tokens": (
                    conversation_event["output_tokens"]
                ),
                "latency_ms": (
                    conversation_event["latency_ms"]
                ),
                "stop_reason": (
                    conversation_event["stop_reason"]
                ),
            },
            "premium": {
                "route": premium_event["route"],
                "requested_capability": (
                    premium_event[
                        "requested_capability"
                    ]
                ),
                "model_calls": (
                    premium_event["model_calls"]
                ),
                "input_tokens": (
                    premium_event["input_tokens"]
                ),
                "output_tokens": (
                    premium_event["output_tokens"]
                ),
                "latency_ms": (
                    premium_event["latency_ms"]
                ),
                "stop_reason": (
                    premium_event["stop_reason"]
                ),
            },
            "same_physical_model": (
                conversation_event["model_id"]
                == premium_event["model_id"]
            ),
            "capability_qualification_verified": False,
            "total_model_calls": 2,
            "external_actions_executed": 0,
            "route_execution_valid": execution_valid,
        },
        indent=2,
    )
)

print("SECRETS PRINTED: False")
print("V2 LAB 5 RUN: COMPLETE")
