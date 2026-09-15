import json
import os
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from anthropic import Anthropic
from dotenv import load_dotenv

from v2.services.context_builder import (
    build_explicit_context,
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

context = build_explicit_context("CASE-001")

client = Anthropic(api_key=api_key)
started = perf_counter()

response = client.messages.create(
    model=model,
    max_tokens=450,
    system=context["system"],
    messages=context["messages"],
)

latency_ms = round(
    (perf_counter() - started) * 1000,
    2,
)

response_text = "\n".join(
    block.text
    for block in response.content
    if block.type == "text"
).strip()

if not response_text:
    raise SystemExit("Claude returned no text.")

usage = response.usage

event = {
    "interaction_id": str(uuid4()),
    "created_at": datetime.now(timezone.utc).isoformat(),
    "lab": "V2_LAB_02",
    "case_id": context["case_id"],
    "route": "DIRECT_EXPLICIT_CONTEXT",
    "context_strategy": "FULL_RECONSTRUCTION",
    "authoritative_sources": [
        "submission",
        "compliance_rules",
    ],
    "component_characters": (
        context["component_characters"]
    ),
    "context_characters": (
        context["wire_context_characters"]
    ),
    "model_id": response.model,
    "model_calls": 1,
    "input_tokens": usage.input_tokens,
    "output_tokens": usage.output_tokens,
    "cache_creation_input_tokens": (
        getattr(
            usage,
            "cache_creation_input_tokens",
            0,
        )
        or 0
    ),
    "cache_read_input_tokens": (
        getattr(
            usage,
            "cache_read_input_tokens",
            0,
        )
        or 0
    ),
    "response_characters": len(response_text),
    "tool_calls": 0,
    "agent_steps": 0,
    "latency_ms": latency_ms,
    "stop_reason": response.stop_reason,
    "business_outcome": "MODEL_RECOMMENDATION_CREATED",
}

record_event(event)

response_upper = response_text.upper()

print("\nCLAUDE RESPONSE:\n")
print(response_text)

print("\nEXPLICIT CONTEXT TELEMETRY:")
print(
    json.dumps(
        {
            "lab": event["lab"],
            "route": event["route"],
            "context_strategy": (
                event["context_strategy"]
            ),
            "authoritative_sources": (
                event["authoritative_sources"]
            ),
            "component_characters": (
                event["component_characters"]
            ),
            "context_characters": (
                event["context_characters"]
            ),
            "input_tokens": event["input_tokens"],
            "output_tokens": event["output_tokens"],
            "latency_ms": event["latency_ms"],
            "tool_calls": event["tool_calls"],
            "stop_reason": event["stop_reason"],
        },
        indent=2,
    )
)

print(
    "\nRECOMMENDATION MENTIONS EVIDENCE:",
    "REQUEST" in response_upper
    and "EVIDENCE" in response_upper,
)
print("SECRETS PRINTED: False")
print("V2 LAB 2 RUN: COMPLETE")
