from collections.abc import Iterable, Mapping
from typing import Any


CONSEQUENTIAL_ACTIONS = frozenset(
    {
        "APPROVE",
        "DECLINE",
        "BIND_COVERAGE",
        "SET_PREMIUM",
    }
)


def is_tool_allowed(
    tool_name: str,
    allowed_tools: Iterable[str],
) -> bool:
    normalized_name = tool_name.strip()
    normalized_allowlist = {
        name.strip()
        for name in allowed_tools
    }
    return normalized_name in normalized_allowlist


def authorize_underwriting_step(
    case_id: str,
    requested_action: str,
    compliance: Mapping[str, Any],
    human_approval_supplied: bool = False,
) -> dict[str, Any]:
    normalized_case_id = case_id.strip().upper()
    policy_case_id = str(
        compliance.get("case_id", "")
    ).strip().upper()
    action = requested_action.strip().upper()
    policy_next_step = str(
        compliance.get("next_step", "")
    ).strip().upper()

    result = {
        "case_id": normalized_case_id,
        "policy_id": compliance.get("policy_id"),
        "requested_action": action,
        "policy_next_step": policy_next_step,
        "human_review_required": (
            compliance.get("human_review_required") is True
        ),
        "human_approval_required": (
            compliance.get("human_approval_required") is True
        ),
        "human_approval_supplied": human_approval_supplied,
        "external_action_executed": False,
        "coverage_decision_executed": False,
    }

    if not policy_case_id or policy_case_id != normalized_case_id:
        return {
            **result,
            "authorization_status": "BLOCKED",
            "reason_code": "CASE_ID_MISMATCH",
        }

    if not policy_next_step:
        return {
            **result,
            "authorization_status": "BLOCKED",
            "reason_code": "MISSING_POLICY_NEXT_STEP",
        }

    if (
        action in CONSEQUENTIAL_ACTIONS
        and result["human_approval_required"]
        and not human_approval_supplied
    ):
        return {
            **result,
            "authorization_status": "BLOCKED",
            "reason_code": "HUMAN_APPROVAL_REQUIRED",
        }

    if action != policy_next_step:
        return {
            **result,
            "authorization_status": "BLOCKED",
            "reason_code": "POLICY_MISMATCH",
        }

    return {
        **result,
        "authorization_status": "PERMITTED",
        "reason_code": "POLICY_CONTROLLED_WORKFLOW_STEP",
    }
