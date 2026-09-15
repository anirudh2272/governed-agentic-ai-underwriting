import json
from typing import Any

from tools.underwriting_tools import (
    compliance_rules,
    submission,
)

SYSTEM_INSTRUCTIONS = (
    "You are an underwriting analysis assistant in a "
    "synthetic training environment. Use only the "
    "provided case and policy state. Missing evidence "
    "is unknown and must not be interpreted as zero "
    "losses. Produce a recommendation only. Never "
    "approve, decline, price, or bind coverage."
)

TASK_INSTRUCTIONS = (
    "Return exactly four sections: CASE_FACTS, "
    "EVIDENCE_GAPS, POLICY_NEXT_STEP, and "
    "AUTHORIZATION_BOUNDARY. Use only values explicitly "
    "present in the supplied case and policy. Copy "
    "categorical values exactly without expanding or "
    "defining them. Use exact evidence names and the exact "
    "policy next_step. Do not add document formats, "
    "lookback periods, industry practices, definitions, "
    "or requirements. If a fact was not supplied, write "
    "UNKNOWN."
)


def _validate_result(
    source_name: str,
    result: dict[str, Any],
) -> None:
    if "error" in result:
        raise ValueError(
            f"{source_name} failed: {result['error']}"
        )


def build_explicit_context(
    case_id: str,
) -> dict[str, Any]:
    """Reconstruct one complete, stateless model request."""

    case_state = submission(case_id)
    policy_state = compliance_rules(case_id)

    _validate_result("submission", case_state)
    _validate_result(
        "compliance_rules",
        policy_state,
    )

    case_json = json.dumps(
        case_state,
        indent=2,
        sort_keys=True,
    )

    policy_json = json.dumps(
        policy_state,
        indent=2,
        sort_keys=True,
    )

    user_prompt = (
        "AUTHORITATIVE CASE STATE:\n"
        f"{case_json}\n\n"
        "AUTHORITATIVE POLICY STATE:\n"
        f"{policy_json}\n\n"
        "CURRENT TASK:\n"
        f"{TASK_INSTRUCTIONS}"
    )

    component_characters = {
        "system_instructions": len(
            SYSTEM_INSTRUCTIONS
        ),
        "case_state": len(case_json),
        "policy_state": len(policy_json),
        "task_instructions": len(
            TASK_INSTRUCTIONS
        ),
    }

    return {
        "case_id": case_id,
        "system": SYSTEM_INSTRUCTIONS,
        "messages": [
            {
                "role": "user",
                "content": user_prompt,
            }
        ],
        "component_characters": (
            component_characters
        ),
        "wire_context_characters": (
            len(SYSTEM_INSTRUCTIONS)
            + len(user_prompt)
        ),
    }
