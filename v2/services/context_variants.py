import json
from typing import Any

from tools.underwriting_tools import (
    compliance_rules,
    submission,
)

COMMON_SYSTEM = (
    "You are a workflow recommendation assistant in a "
    "synthetic underwriting environment. Use only the "
    "supplied authoritative state. Do not add facts, "
    "requirements, definitions, or decisions. A human "
    "owns consequential underwriting decisions."
)

COMMON_TASK = (
    "State the case ID, evidence gaps, exact policy "
    "next_step, human-review flag, and human-approval "
    "flag. Recommend workflow progression only."
)


def _build_variant(
    strategy: str,
    case_state: dict[str, Any],
    policy_state: dict[str, Any],
) -> dict[str, Any]:
    payload = {
        "case_state": case_state,
        "policy_state": policy_state,
    }

    payload_json = json.dumps(
        payload,
        indent=2,
        sort_keys=True,
    )

    user_prompt = (
        "AUTHORITATIVE CONTEXT:\n"
        f"{payload_json}\n\n"
        "CURRENT TASK:\n"
        f"{COMMON_TASK}"
    )

    return {
        "strategy": strategy,
        "system": COMMON_SYSTEM,
        "messages": [
            {
                "role": "user",
                "content": user_prompt,
            }
        ],
        "payload": payload,
        "payload_characters": len(payload_json),
        "wire_context_characters": (
            len(COMMON_SYSTEM) + len(user_prompt)
        ),
    }


def build_context_variants(
    case_id: str,
) -> dict[str, Any]:
    case_state = submission(case_id)
    policy_state = compliance_rules(case_id)

    if "error" in case_state:
        raise ValueError(case_state["error"])

    if "error" in policy_state:
        raise ValueError(policy_state["error"])

    essential_case_fields = {
        "case_id",
        "required_evidence",
        "outstanding_evidence",
        "evidence_status",
    }

    essential_policy_fields = {
        "policy_id",
        "next_step",
        "human_review_required",
        "human_approval_required",
    }

    reduced_case = {
        key: case_state[key]
        for key in essential_case_fields
    }

    reduced_policy = {
        key: policy_state[key]
        for key in essential_policy_fields
    }

    full = _build_variant(
        "FULL_CONTEXT",
        case_state,
        policy_state,
    )

    reduced = _build_variant(
        "REDUCED_CONTEXT",
        reduced_case,
        reduced_policy,
    )

    return {
        "full": full,
        "reduced": reduced,
        "removed_case_fields": sorted(
            set(case_state) - essential_case_fields
        ),
        "removed_policy_fields": sorted(
            set(policy_state) - essential_policy_fields
        ),
    }
