from collections.abc import Iterable
from typing import Any

from tools.underwriting_controls import is_tool_allowed
from tools.underwriting_tools import (
    base_score,
    compliance_rules,
    loss_history,
    submission,
    weather_risk,
)

TOOL_REGISTRY = {
    "submission": submission,
    "weather_risk": weather_risk,
    "compliance_rules": compliance_rules,
    "loss_history": loss_history,
    "base_score": base_score,
}

TOOL_ARGUMENTS = {
    "submission": frozenset({"case_id"}),
    "weather_risk": frozenset({"location"}),
    "compliance_rules": frozenset({"case_id"}),
    "loss_history": frozenset({"case_id"}),
    "base_score": frozenset({"case_id"}),
}

DEFAULT_ALLOWED_TOOLS = frozenset(
    TOOL_REGISTRY
)


def execute_governed_tool(
    tool_name: str,
    arguments: dict[str, Any],
    *,
    allowed_tools: Iterable[str] = (
        DEFAULT_ALLOWED_TOOLS
    ),
) -> dict[str, Any]:
    """Deny by default, validate arguments, then execute."""

    if not is_tool_allowed(
        tool_name,
        allowed_tools,
    ):
        return {
            "tool_name": tool_name,
            "status": "BLOCKED",
            "executed": False,
            "reason_code": "TOOL_NOT_ALLOWED",
            "result": None,
            "external_action_executed": False,
        }

    if tool_name not in TOOL_REGISTRY:
        return {
            "tool_name": tool_name,
            "status": "BLOCKED",
            "executed": False,
            "reason_code": "TOOL_NOT_REGISTERED",
            "result": None,
            "external_action_executed": False,
        }

    expected_arguments = TOOL_ARGUMENTS[tool_name]

    if set(arguments) != expected_arguments:
        return {
            "tool_name": tool_name,
            "status": "BLOCKED",
            "executed": False,
            "reason_code": "INVALID_ARGUMENTS",
            "result": None,
            "external_action_executed": False,
        }

    if any(
        not isinstance(value, str)
        or not value.strip()
        for value in arguments.values()
    ):
        return {
            "tool_name": tool_name,
            "status": "BLOCKED",
            "executed": False,
            "reason_code": "INVALID_ARGUMENT_VALUES",
            "result": None,
            "external_action_executed": False,
        }

    try:
        result = TOOL_REGISTRY[tool_name](
            **arguments
        )
    except Exception as error:
        return {
            "tool_name": tool_name,
            "status": "ERROR",
            "executed": False,
            "reason_code": "TOOL_EXECUTION_ERROR",
            "error_type": type(error).__name__,
            "result": None,
            "external_action_executed": False,
        }

    if isinstance(result, dict) and "error" in result:
        return {
            "tool_name": tool_name,
            "status": "ERROR",
            "executed": True,
            "reason_code": "TOOL_RETURNED_ERROR",
            "result": result,
            "external_action_executed": False,
        }

    return {
        "tool_name": tool_name,
        "status": "SUCCESS",
        "executed": True,
        "reason_code": "READ_ONLY_TOOL_EXECUTED",
        "result": result,
        "external_action_executed": False,
    }
