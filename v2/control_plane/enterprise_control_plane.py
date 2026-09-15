"""Unified Enterprise AI Control Plane V2.

Coordinates routing, data policy, model selection, tool permissions,
cost limits, application authorization, governed execution, and
telemetry.

This module implements synthetic training controls only.
"""

from collections.abc import Iterable, Mapping
from typing import Any

from tools.underwriting_controls import (
    authorize_underwriting_step,
)
from v2.control_plane.model_selector import (
    load_model_catalog,
    select_model,
)
from v2.control_plane.policies import (
    VALID_DATA_CLASSIFICATIONS,
    get_route_policy,
    is_data_classification_allowed,
)
from v2.control_plane.router import route_request
from v2.control_plane.tool_gateway import (
    execute_governed_tool,
)
from v2.services.context_builder import (
    build_explicit_context,
)
from v2.services.route_executor import execute_route
from v2.services.telemetry import record_event


def _finish(
    result: dict[str, Any],
    status: str,
    reason_code: str,
) -> dict[str, Any]:
    """Finalize the result and write centralized telemetry."""

    result["control_plane_status"] = status
    result["reason_code"] = reason_code

    selection = result.get("model_selection") or {}
    execution = result.get("execution") or {}
    authorization = result.get("authorization") or {}

    result["external_action_executed"] = bool(
        authorization.get(
            "external_action_executed",
            False,
        )
    )
    result["coverage_decision_executed"] = bool(
        authorization.get(
            "coverage_decision_executed",
            False,
        )
    )

    record_event(
        {
            "lab": "V2_LAB_11",
            "event_type": (
                "ENTERPRISE_CONTROL_PLANE_EXECUTION"
            ),
            "case_id": result.get("case_id"),
            "control_plane_status": status,
            "reason_code": reason_code,
            "data_classification": result.get(
                "data_classification"
            ),
            "route": result.get("route"),
            "model_id": selection.get("model_id"),
            "model_selection_status": selection.get(
                "selection_status"
            ),
            "estimated_model_cost_usd": (
                selection.get(
                    "estimated_model_cost_usd"
                )
                or 0.0
            ),
            "model_calls": execution.get(
                "model_calls",
                0,
            ),
            "input_tokens": execution.get(
                "input_tokens",
                0,
            ),
            "output_tokens": execution.get(
                "output_tokens",
                0,
            ),
            "latency_ms": execution.get(
                "latency_ms",
                0.0,
            ),
            "requested_tools": result.get(
                "requested_tools",
                [],
            ),
            "model_tools_exposed": result.get(
                "model_tools_exposed",
                [],
            ),
            "authorization_status": (
                authorization.get(
                    "authorization_status"
                )
            ),
            "external_action_executed": result[
                "external_action_executed"
            ],
            "coverage_decision_executed": result[
                "coverage_decision_executed"
            ],
        }
    )

    return result


def run_control_plane(
    *,
    case_id: str,
    task_type: str,
    deterministic_answer: str | None,
    complexity: str,
    business_risk: str,
    data_classification: str,
    requested_action: str | None = None,
    requested_tools: Iterable[str] = (),
    requested_agent_steps: int = 0,
    expected_input_tokens: int = 0,
    expected_output_tokens: int = 0,
    human_approval_supplied: bool = False,
    model_consequential_authority_requested: bool = False,
    catalog_entries: (
        Iterable[Mapping[str, Any]] | None
    ) = None,
    client: Any = None,
) -> dict[str, Any]:
    """Process one request through the governed control plane."""

    normalized_case_id = (
        case_id.strip().upper()
        if isinstance(case_id, str)
        else ""
    )
    normalized_classification = (
        data_classification.strip().upper()
        if isinstance(data_classification, str)
        else ""
    )
    normalized_action = (
        requested_action.strip().upper()
        if isinstance(requested_action, str)
        and requested_action.strip()
        else None
    )

    result: dict[str, Any] = {
        "case_id": normalized_case_id,
        "task_type": task_type,
        "data_classification": (
            normalized_classification
        ),
        "requested_action": normalized_action,
        "requested_tools": [],
        "requested_agent_steps": (
            requested_agent_steps
        ),
        "route": None,
        "route_decision": None,
        "route_policy": None,
        "model_selection": None,
        "model_tools_exposed": [],
        "policy_tool_call": None,
        "authorization": None,
        "execution": None,
        "external_action_executed": False,
        "coverage_decision_executed": False,
    }

    if not normalized_case_id or not isinstance(
        task_type,
        str,
    ) or not task_type.strip():
        return _finish(
            result,
            "INVALID_REQUEST",
            "MISSING_REQUIRED_REQUEST_FIELD",
        )

    if normalized_classification not in (
        VALID_DATA_CLASSIFICATIONS
    ):
        return _finish(
            result,
            "INVALID_REQUEST",
            "INVALID_DATA_CLASSIFICATION",
        )

    numeric_limits = (
        requested_agent_steps,
        expected_input_tokens,
        expected_output_tokens,
    )

    if any(
        type(value) is not int or value < 0
        for value in numeric_limits
    ):
        return _finish(
            result,
            "INVALID_REQUEST",
            "INVALID_NUMERIC_LIMIT",
        )

    try:
        requested_tool_names = tuple(requested_tools)
    except TypeError:
        return _finish(
            result,
            "INVALID_REQUEST",
            "INVALID_TOOL_LIST",
        )

    if isinstance(requested_tools, (str, bytes)) or any(
        not isinstance(name, str)
        or not name.strip()
        for name in requested_tool_names
    ):
        return _finish(
            result,
            "INVALID_REQUEST",
            "INVALID_TOOL_LIST",
        )

    normalized_tools = frozenset(
        name.strip()
        for name in requested_tool_names
    )
    result["requested_tools"] = sorted(
        normalized_tools
    )

    try:
        decision = route_request(
            task_type=task_type.strip(),
            deterministic_answer=(
                deterministic_answer
            ),
            complexity=complexity,
            business_risk=business_risk,
        )
    except (TypeError, ValueError):
        return _finish(
            result,
            "INVALID_REQUEST",
            "ROUTER_INPUT_REJECTED",
        )

    route = decision["route"]
    result["route"] = route
    result["route_decision"] = decision

    policy = get_route_policy(route)
    allowed_tools = policy["allowed_tools"]

    result["route_policy"] = {
        "route": policy["route"],
        "allowed_data_classifications": sorted(
            policy[
                "allowed_data_classifications"
            ]
        ),
        "allowed_tools": sorted(allowed_tools),
        "max_agent_steps": policy[
            "max_agent_steps"
        ],
        "max_estimated_model_cost_usd": policy[
            "max_estimated_model_cost_usd"
        ],
    }

    if not is_data_classification_allowed(
        route,
        normalized_classification,
    ):
        return _finish(
            result,
            "BLOCKED",
            "DATA_CLASSIFICATION_NOT_ALLOWED",
        )

    prohibited_tools = (
        normalized_tools - allowed_tools
    )

    if prohibited_tools:
        result["prohibited_tools"] = sorted(
            prohibited_tools
        )
        return _finish(
            result,
            "BLOCKED",
            "TOOL_NOT_ALLOWED_FOR_ROUTE",
        )

    if requested_agent_steps > policy[
        "max_agent_steps"
    ]:
        return _finish(
            result,
            "BLOCKED",
            "AGENT_STEP_LIMIT_EXCEEDED",
        )

    entries = (
        list(catalog_entries)
        if catalog_entries is not None
        else load_model_catalog()
    )

    selection = select_model(
        route,
        entries,
        expected_input_tokens=(
            expected_input_tokens
        ),
        expected_output_tokens=(
            expected_output_tokens
        ),
        consequential_action_requested=(
            model_consequential_authority_requested
        ),
    )
    result["model_selection"] = selection

    if selection["selection_status"] != "SELECTED":
        return _finish(
            result,
            "BLOCKED",
            "MODEL_SELECTION_FAILED",
        )

    estimated_cost = float(
        selection["estimated_model_cost_usd"]
    )
    cost_limit = float(
        policy["max_estimated_model_cost_usd"]
    )

    if estimated_cost > cost_limit:
        return _finish(
            result,
            "BLOCKED",
            "ESTIMATED_COST_LIMIT_EXCEEDED",
        )

    result["model_tools_exposed"] = sorted(
        normalized_tools
    )

    if normalized_action is not None:
        policy_call = execute_governed_tool(
            "compliance_rules",
            {"case_id": normalized_case_id},
            allowed_tools={"compliance_rules"},
        )
        result["policy_tool_call"] = policy_call

        compliance = policy_call.get("result")

        if (
            policy_call.get("executed") is not True
            or not isinstance(compliance, dict)
        ):
            return _finish(
                result,
                "BLOCKED",
                "POLICY_LOOKUP_FAILED",
            )

        authorization = (
            authorize_underwriting_step(
                normalized_case_id,
                normalized_action,
                compliance,
                human_approval_supplied=(
                    human_approval_supplied
                ),
            )
        )
        result["authorization"] = authorization

        if (
            authorization.get(
                "authorization_status"
            )
            != "PERMITTED"
        ):
            return _finish(
                result,
                "BLOCKED",
                authorization.get(
                    "reason_code",
                    "APPLICATION_AUTHORIZATION_BLOCKED",
                ),
            )
    else:
        result["authorization"] = {
            "authorization_status": (
                "NOT_REQUESTED"
            ),
            "external_action_executed": False,
            "coverage_decision_executed": False,
        }

    if route == "NO_LLM":
        execution = execute_route(decision)
    else:
        if client is None:
            return _finish(
                result,
                "BLOCKED",
                "MODEL_CLIENT_REQUIRED",
            )

        context = build_explicit_context(
            normalized_case_id
        )

        result["context_characters"] = context[
            "wire_context_characters"
        ]

        execution = execute_route(
            decision,
            client=client,
            model_id=selection["model_id"],
            system=context["system"],
            messages=context["messages"],
            max_tokens=(
                expected_output_tokens
                if expected_output_tokens > 0
                else 300
            ),
        )

    result["execution"] = execution

    return _finish(
        result,
        "COMPLETED",
        "GOVERNED_EXECUTION_COMPLETED",
    )
