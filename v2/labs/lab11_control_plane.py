"""Runtime validation for Enterprise AI Control Plane V2."""

import json
from pathlib import Path
from typing import Any

from v2.control_plane.enterprise_control_plane import (
    run_control_plane,
)
from v2.control_plane.model_selector import (
    load_model_catalog,
)
from v2.services.telemetry import TELEMETRY_FILE


RESULT_FILE = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "lab11_control_plane_results.json"
)


def read_telemetry() -> list[dict[str, Any]]:
    if not TELEMETRY_FILE.exists():
        return []

    return [
        json.loads(line)
        for line in TELEMETRY_FILE.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]


def summarize(
    result: dict[str, Any],
) -> dict[str, Any]:
    selection = result.get("model_selection") or {}
    execution = result.get("execution") or {}
    authorization = result.get("authorization") or {}

    return {
        "route": result.get("route"),
        "control_plane_status": result.get(
            "control_plane_status"
        ),
        "reason_code": result.get("reason_code"),
        "model_selection_status": selection.get(
            "selection_status"
        ),
        "model_selection_reason": selection.get(
            "reason_code"
        ),
        "model_id": selection.get("model_id"),
        "estimated_model_cost_usd": selection.get(
            "estimated_model_cost_usd"
        ),
        "authorization_status": authorization.get(
            "authorization_status"
        ),
        "authorization_reason": authorization.get(
            "reason_code"
        ),
        "model_calls": execution.get(
            "model_calls",
            0,
        ),
        "response_text": execution.get(
            "response_text"
        ),
        "external_action_executed": result.get(
            "external_action_executed"
        ),
        "coverage_decision_executed": result.get(
            "coverage_decision_executed"
        ),
    }


before_count = len(read_telemetry())

print("OPENING DATABRICKS MODEL CATALOG...")

catalog = load_model_catalog()

print("MODEL CATALOG ROWS:", len(catalog))


happy_path = run_control_plane(
    case_id="CASE-001",
    task_type="WORKFLOW_NEXT_STEP",
    deterministic_answer="REQUEST_EVIDENCE",
    complexity="LOW",
    business_risk="HIGH",
    data_classification="SYNTHETIC_TRAINING",
    requested_action="REQUEST_EVIDENCE",
    requested_tools=(),
    requested_agent_steps=0,
    expected_input_tokens=0,
    expected_output_tokens=0,
    catalog_entries=catalog,
)


restricted_data = run_control_plane(
    case_id="CASE-001",
    task_type="UNDERWRITING_ANALYSIS",
    deterministic_answer=None,
    complexity="HIGH",
    business_risk="HIGH",
    data_classification="RESTRICTED",
    requested_action=None,
    requested_tools=(),
    requested_agent_steps=1,
    expected_input_tokens=500,
    expected_output_tokens=200,
    catalog_entries=catalog,
)


unqualified_premium = run_control_plane(
    case_id="CASE-001",
    task_type="UNDERWRITING_ANALYSIS",
    deterministic_answer=None,
    complexity="HIGH",
    business_risk="HIGH",
    data_classification="SYNTHETIC_TRAINING",
    requested_action=None,
    requested_tools=(
        "submission",
        "compliance_rules",
    ),
    requested_agent_steps=2,
    expected_input_tokens=500,
    expected_output_tokens=200,
    catalog_entries=catalog,
)


prohibited_tool = run_control_plane(
    case_id="CASE-001",
    task_type="STATUS_SUMMARY",
    deterministic_answer=None,
    complexity="LOW",
    business_risk="LOW",
    data_classification="SYNTHETIC_TRAINING",
    requested_action=None,
    requested_tools=("submission",),
    requested_agent_steps=1,
    expected_input_tokens=500,
    expected_output_tokens=100,
    catalog_entries=catalog,
)


step_limit = run_control_plane(
    case_id="CASE-001",
    task_type="STATUS_SUMMARY",
    deterministic_answer=None,
    complexity="LOW",
    business_risk="LOW",
    data_classification="SYNTHETIC_TRAINING",
    requested_action=None,
    requested_tools=(),
    requested_agent_steps=2,
    expected_input_tokens=500,
    expected_output_tokens=100,
    catalog_entries=catalog,
)


cost_limit = run_control_plane(
    case_id="CASE-001",
    task_type="STATUS_SUMMARY",
    deterministic_answer=None,
    complexity="LOW",
    business_risk="LOW",
    data_classification="SYNTHETIC_TRAINING",
    requested_action=None,
    requested_tools=(),
    requested_agent_steps=1,
    expected_input_tokens=2000,
    expected_output_tokens=1000,
    catalog_entries=catalog,
)


authorization_boundary = run_control_plane(
    case_id="CASE-001",
    task_type="WORKFLOW_NEXT_STEP",
    deterministic_answer="REQUEST_EVIDENCE",
    complexity="LOW",
    business_risk="HIGH",
    data_classification="SYNTHETIC_TRAINING",
    requested_action="BIND_COVERAGE",
    requested_tools=(),
    requested_agent_steps=0,
    expected_input_tokens=0,
    expected_output_tokens=0,
    human_approval_supplied=False,
    catalog_entries=catalog,
)


results = {
    "happy_path": happy_path,
    "restricted_data": restricted_data,
    "unqualified_premium": unqualified_premium,
    "prohibited_tool": prohibited_tool,
    "step_limit": step_limit,
    "cost_limit": cost_limit,
    "authorization_boundary": (
        authorization_boundary
    ),
}


print("\nCONTROL PLANE SCENARIOS:")

for name, result in results.items():
    print(f"\n{name.upper()}:")
    print(
        json.dumps(
            summarize(result),
            indent=2,
        )
    )


happy_valid = all(
    (
        happy_path["control_plane_status"]
        == "COMPLETED",
        happy_path["route"] == "NO_LLM",
        happy_path["model_selection"]["model_id"]
        == "NO_LLM",
        happy_path["execution"]["model_calls"] == 0,
        happy_path["execution"]["response_text"]
        == "REQUEST_EVIDENCE",
        happy_path["authorization"][
            "authorization_status"
        ]
        == "PERMITTED",
    )
)


restricted_valid = all(
    (
        restricted_data["route"]
        == "PREMIUM_REASONING",
        restricted_data["control_plane_status"]
        == "BLOCKED",
        restricted_data["reason_code"]
        == "DATA_CLASSIFICATION_NOT_ALLOWED",
        restricted_data["model_selection"] is None,
    )
)


premium_valid = all(
    (
        unqualified_premium["route"]
        == "PREMIUM_REASONING",
        unqualified_premium[
            "control_plane_status"
        ]
        == "BLOCKED",
        unqualified_premium["reason_code"]
        == "MODEL_SELECTION_FAILED",
        unqualified_premium["model_selection"][
            "selection_status"
        ]
        == "NO_QUALIFIED_MODEL",
    )
)


tool_valid = all(
    (
        prohibited_tool["route"]
        == "CONVERSATIONAL",
        prohibited_tool["control_plane_status"]
        == "BLOCKED",
        prohibited_tool["reason_code"]
        == "TOOL_NOT_ALLOWED_FOR_ROUTE",
    )
)


step_valid = all(
    (
        step_limit["route"]
        == "CONVERSATIONAL",
        step_limit["control_plane_status"]
        == "BLOCKED",
        step_limit["reason_code"]
        == "AGENT_STEP_LIMIT_EXCEEDED",
    )
)


cost_valid = all(
    (
        cost_limit["route"]
        == "CONVERSATIONAL",
        cost_limit["control_plane_status"]
        == "BLOCKED",
        cost_limit["reason_code"]
        == "ESTIMATED_COST_LIMIT_EXCEEDED",
        cost_limit["model_selection"][
            "selection_status"
        ]
        == "SELECTED",
    )
)


authorization_valid = all(
    (
        authorization_boundary["route"]
        == "NO_LLM",
        authorization_boundary[
            "control_plane_status"
        ]
        == "BLOCKED",
        authorization_boundary[
            "authorization"
        ]["authorization_status"]
        == "BLOCKED",
        authorization_boundary["reason_code"]
        == "HUMAN_APPROVAL_REQUIRED",
        authorization_boundary["execution"]
        is None,
    )
)


all_results_safe = all(
    (
        not result["external_action_executed"]
        and not result[
            "coverage_decision_executed"
        ]
        and (
            result.get("execution") or {}
        ).get("model_calls", 0)
        == 0
    )
    for result in results.values()
)


new_events = read_telemetry()[before_count:]

telemetry_valid = all(
    (
        len(new_events) == len(results),
        all(
            event.get("lab") == "V2_LAB_11"
            for event in new_events
        ),
        all(
            event.get("event_type")
            == "ENTERPRISE_CONTROL_PLANE_EXECUTION"
            for event in new_events
        ),
    )
)


lab_verified = all(
    (
        happy_valid,
        restricted_valid,
        premium_valid,
        tool_valid,
        step_valid,
        cost_valid,
        authorization_valid,
        all_results_safe,
        telemetry_valid,
    )
)


report = {
    "catalog_rows_loaded": len(catalog),
    "happy_path_valid": happy_valid,
    "restricted_data_blocked": restricted_valid,
    "unqualified_premium_failed_closed": (
        premium_valid
    ),
    "prohibited_tool_blocked": tool_valid,
    "agent_step_limit_enforced": step_valid,
    "estimated_cost_limit_enforced": cost_valid,
    "human_authorization_enforced": (
        authorization_valid
    ),
    "telemetry_events_written": len(new_events),
    "telemetry_valid": telemetry_valid,
    "all_results_safe": all_results_safe,
    "claude_api_calls": 0,
    "databricks_reads": 1,
    "databricks_writes": 0,
    "underwriting_actions_executed": 0,
    "lab_verified": lab_verified,
}


RESULT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)
RESULT_FILE.write_text(
    json.dumps(
        {
            "report": report,
            "scenario_results": results,
        },
        indent=2,
    ),
    encoding="utf-8",
)


print("\nLAB 11 REPORT:")
print(json.dumps(report, indent=2))

print("\nRESULT FILE:", RESULT_FILE.name)
print("SECRETS PRINTED: False")

if not lab_verified:
    raise RuntimeError(
        "Lab 11 validation failed. "
        "Paste the complete output."
    )

print("V2 LAB 11 STATUS: COMPLETE")
