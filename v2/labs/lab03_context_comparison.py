import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from anthropic import Anthropic
from dotenv import load_dotenv

from v2.services.context_variants import (
    build_context_variants,
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

client = Anthropic(api_key=api_key)
variants = build_context_variants("CASE-001")
experiment_id = str(uuid4())


def evaluate_outcome(text: str) -> dict[str, bool]:
    canonical = re.sub(
        r"[^A-Z0-9]+",
        "_",
        text.upper(),
    )

    review_true = bool(
        re.search(
            r"HUMAN[\s_-]*REVIEW.{0,40}"
            r"\b(TRUE|REQUIRED|YES)\b",
            text,
            re.IGNORECASE | re.DOTALL,
        )
    )

    approval_true = bool(
        re.search(
            r"HUMAN[\s_-]*APPROVAL.{0,40}"
            r"\b(TRUE|REQUIRED|YES)\b",
            text,
            re.IGNORECASE | re.DOTALL,
        )
    )

    return {
        "case_id": "CASE_001" in canonical,
        "wind_mitigation": (
            "WIND_MITIGATION" in canonical
        ),
        "contractor_loss_history": (
            "CONTRACTOR_LOSS_HISTORY" in canonical
        ),
        "next_step": (
            "REQUEST_EVIDENCE" in canonical
        ),
        "human_review_true": review_true,
        "human_approval_true": approval_true,
    }


def run_variant(
    variant_name: str,
    context: dict,
) -> dict:
    started = perf_counter()

    response = client.messages.create(
        model=model,
        max_tokens=300,
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
        raise RuntimeError(
            f"{variant_name} returned no text."
        )

    checks = evaluate_outcome(response_text)
    outcome_valid = all(checks.values())
    usage = response.usage

    event = {
        "interaction_id": str(uuid4()),
        "experiment_id": experiment_id,
        "created_at": (
            datetime.now(timezone.utc).isoformat()
        ),
        "lab": "V2_LAB_03",
        "case_id": "CASE-001",
        "route": "DIRECT_CONTEXT_COMPARISON",
        "context_strategy": (
            context["strategy"]
        ),
        "context_characters": (
            context["wire_context_characters"]
        ),
        "model_id": response.model,
        "model_calls": 1,
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "response_characters": len(response_text),
        "tool_calls": 0,
        "agent_steps": 0,
        "latency_ms": latency_ms,
        "stop_reason": response.stop_reason,
        "outcome_checks": checks,
        "outcome_valid": outcome_valid,
        "business_outcome": (
            "WORKFLOW_RECOMMENDATION_CREATED"
        ),
    }

    record_event(event)

    return {
        "name": variant_name,
        "text": response_text,
        "event": event,
    }


full_result = run_variant(
    "FULL",
    variants["full"],
)

reduced_result = run_variant(
    "REDUCED",
    variants["reduced"],
)

full_event = full_result["event"]
reduced_event = reduced_result["event"]

input_saved = (
    full_event["input_tokens"]
    - reduced_event["input_tokens"]
)

input_reduction_percent = round(
    input_saved
    / full_event["input_tokens"]
    * 100,
    2,
)

print("\nFULL CONTEXT RESPONSE:\n")
print(full_result["text"])

print("\nREDUCED CONTEXT RESPONSE:\n")
print(reduced_result["text"])

print("\nCOMPARISON:")
print(
    json.dumps(
        {
            "full_context_characters": (
                full_event["context_characters"]
            ),
            "reduced_context_characters": (
                reduced_event["context_characters"]
            ),
            "full_input_tokens": (
                full_event["input_tokens"]
            ),
            "reduced_input_tokens": (
                reduced_event["input_tokens"]
            ),
            "input_tokens_saved": input_saved,
            "input_token_reduction_percent": (
                input_reduction_percent
            ),
            "full_output_tokens": (
                full_event["output_tokens"]
            ),
            "reduced_output_tokens": (
                reduced_event["output_tokens"]
            ),
            "full_latency_ms": (
                full_event["latency_ms"]
            ),
            "reduced_latency_ms": (
                reduced_event["latency_ms"]
            ),
            "full_outcome_checks": (
                full_event["outcome_checks"]
            ),
            "reduced_outcome_checks": (
                reduced_event["outcome_checks"]
            ),
            "both_outcomes_valid": (
                full_event["outcome_valid"]
                and reduced_event["outcome_valid"]
            ),
            "model_calls": 2,
            "tool_calls": 0,
        },
        indent=2,
    )
)

print("SECRETS PRINTED: False")
print("V2 LAB 3 RUN: COMPLETE")
