from typing import Any

VALID_LEVELS = {
    "LOW",
    "MEDIUM",
    "HIGH",
}


def route_request(
    *,
    task_type: str,
    deterministic_answer: str | None,
    complexity: str,
    business_risk: str,
) -> dict[str, Any]:
    """Select an execution route without calling a model."""

    normalized_task = task_type.strip().upper()
    normalized_complexity = complexity.strip().upper()
    normalized_risk = business_risk.strip().upper()

    if not normalized_task:
        raise ValueError("task_type cannot be blank.")

    if normalized_complexity not in VALID_LEVELS:
        raise ValueError(
            "complexity must be LOW, MEDIUM, or HIGH."
        )

    if normalized_risk not in VALID_LEVELS:
        raise ValueError(
            "business_risk must be LOW, MEDIUM, or HIGH."
        )

    answer = (
        deterministic_answer.strip()
        if isinstance(deterministic_answer, str)
        else None
    )

    if answer:
        return {
            "task_type": normalized_task,
            "route": "NO_LLM",
            "model_required": False,
            "requested_capability": None,
            "deterministic_result": answer,
            "reason_code": (
                "AUTHORITATIVE_ANSWER_AVAILABLE"
            ),
        }

    if (
        normalized_complexity == "HIGH"
        or normalized_risk == "HIGH"
    ):
        return {
            "task_type": normalized_task,
            "route": "PREMIUM_REASONING",
            "model_required": True,
            "requested_capability": (
                "PREMIUM_REASONING"
            ),
            "deterministic_result": None,
            "reason_code": (
                "HIGH_COMPLEXITY_OR_BUSINESS_RISK"
            ),
        }

    return {
        "task_type": normalized_task,
        "route": "CONVERSATIONAL",
        "model_required": True,
        "requested_capability": "CONVERSATIONAL",
        "deterministic_result": None,
        "reason_code": (
            "STANDARD_LANGUAGE_TASK"
        ),
    }
