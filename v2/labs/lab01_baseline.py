import json
import os
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from anthropic import Anthropic
from dotenv import load_dotenv

from tools.underwriting_tools import submission
from v2.services.telemetry import (
    TELEMETRY_FILE,
    record_event,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
model = os.getenv("CLAUDE_MODEL", "").strip()

if not api_key:
    raise SystemExit("ANTHROPIC_API_KEY is missing.")

if not model:
    raise SystemExit("CLAUDE_MODEL is missing.")

case_id = "CASE-001"
case_context = submission(case_id)

if "error" in case_context:
    raise SystemExit(case_context["error"])

context_json = json.dumps(
    case_context,
    indent=2,
)

system_prompt = (
    "You are an underwriting analysis assistant in a "
    "synthetic training environment. Use only supplied "
    "facts. Missing evidence is unknown and must never "
    "be treated as proof of zero losses. Recommend the "
    "next workflow step, but do not approve, decline, "
    "price, or bind coverage."
)

user_prompt = (
    "Review the following synthetic underwriting case. "
    "Return a concise case summary, risk observations, "
    "evidence gaps, and recommended next workflow step.\n\n"
    f"CASE CONTEXT:\n{context_json}"
)

client = Anthropic(api_key=api_key)

started = perf_counter()

response = client.messages.create(
    model=model,
    max_tokens=450,
    system=system_prompt,
    messages=[
        {
            "role": "user",
            "content": user_prompt,
        }
    ],
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
    "lab": "V2_LAB_01",
    "case_id": case_id,
    "route": "DIRECT_BASELINE",
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
    "context_characters": len(context_json),
    "response_characters": len(response_text),
    "tool_calls": 0,
    "agent_steps": 0,
    "latency_ms": latency_ms,
    "stop_reason": response.stop_reason,
    "business_outcome": "MODEL_RECOMMENDATION_CREATED",
}

record_event(event)

print("\nCLAUDE RESPONSE:\n")
print(response_text)

print("\nBASELINE TELEMETRY:")
print(
    json.dumps(
        {
            "case_id": event["case_id"],
            "route": event["route"],
            "model_id": event["model_id"],
            "model_calls": event["model_calls"],
            "input_tokens": event["input_tokens"],
            "output_tokens": event["output_tokens"],
            "context_characters": (
                event["context_characters"]
            ),
            "tool_calls": event["tool_calls"],
            "latency_ms": event["latency_ms"],
            "stop_reason": event["stop_reason"],
            "business_outcome": (
                event["business_outcome"]
            ),
        },
        indent=2,
    )
)

print(
    "\nTELEMETRY FILE:",
    TELEMETRY_FILE.name,
)
print("SECRETS PRINTED: False")
print("V2 LAB 1 RUN: COMPLETE")
