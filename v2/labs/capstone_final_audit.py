import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import anyio
from databricks import sql
from dotenv import load_dotenv
from mcp import Client
from mcp.types import TextContent

from v2.mcp_server.underwriting_server import mcp
from v2.services.telemetry import (
    TELEMETRY_FILE,
    record_event,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "underwriting_cases.json"
)
ORIGINAL_BACKUP_FILE = (
    PROJECT_ROOT
    / "v2"
    / "data"
    / "capstone_underwriting_cases_backup.json"
)
PRE_HUMAN_BACKUP_FILE = (
    PROJECT_ROOT
    / "v2"
    / "data"
    / "capstone_pre_human_decision_backup.json"
)
STATE_TRANSITION_FILE = (
    PROJECT_ROOT
    / "v2"
    / "data"
    / "capstone_state_transition.json"
)
QUALIFICATION_FILE = (
    PROJECT_ROOT
    / "v2"
    / "data"
    / "capstone_model_qualification.json"
)
PREFLIGHT_FILE = (
    PROJECT_ROOT
    / "v2"
    / "data"
    / "capstone_phase_b_preflight.json"
)
PHASE_B_FILE = (
    PROJECT_ROOT
    / "v2"
    / "data"
    / "capstone_phase_b_recommendation.json"
)
HUMAN_FILE = (
    PROJECT_ROOT
    / "v2"
    / "data"
    / "capstone_human_decision.json"
)
DATABRICKS_FILE = (
    PROJECT_ROOT
    / "v2"
    / "data"
    / "capstone_databricks_persistence.json"
)
LAB11_FILE = (
    PROJECT_ROOT
    / "v2"
    / "data"
    / "lab11_control_plane_results.json"
)
FINAL_REPORT_FILE = (
    PROJECT_ROOT
    / "v2"
    / "data"
    / "capstone_final_report.json"
)

CASE_ID = "CASE-001"
DECISION_ID = "CAPSTONE-CASE001-DECISION-001"
LEDGER_EVENT_ID = "CAPSTONE-CASE001-PHASEB-001"
FINAL_EVENT_ID = "CAPSTONE-CASE001-FINAL-AUDIT-001"
MODEL_ID = "claude-haiku-4-5-20251001"

EXPECTED_TOOLS = {
    "submission",
    "weather_risk",
    "compliance_rules",
    "loss_history",
    "base_score",
}

load_dotenv(PROJECT_ROOT / ".env")


def load_json(path: Path) -> dict:
    value = json.loads(
        path.read_text(encoding="utf-8")
    )

    if not isinstance(value, dict):
        raise RuntimeError(
            f"{path.name} must contain a JSON object"
        )

    return value


def load_events() -> list[dict]:
    if not TELEMETRY_FILE.exists():
        return []

    events = []

    for line in TELEMETRY_FILE.read_text(
        encoding="utf-8"
    ).splitlines():
        if not line.strip():
            continue

        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue

        if isinstance(value, dict):
            events.append(value)

    return events


def write_json_atomic(
    path: Path,
    payload: dict,
) -> None:
    temporary = path.with_name(
        path.name + ".tmp"
    )
    temporary.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def collect_key_values(
    value: Any,
    target_key: str,
) -> list[Any]:
    matches = []

    if isinstance(value, dict):
        for key, child in value.items():
            if key == target_key:
                matches.append(child)

            matches.extend(
                collect_key_values(
                    child,
                    target_key,
                )
            )

    elif isinstance(value, list):
        for child in value:
            matches.extend(
                collect_key_values(
                    child,
                    target_key,
                )
            )

    return matches


def any_true(
    value: Any,
    *candidate_keys: str,
) -> bool:
    return any(
        item is True
        for key in candidate_keys
        for item in collect_key_values(
            value,
            key,
        )
    )


def outstanding_evidence(case: dict) -> list[str]:
    required = case.get(
        "required_evidence",
        [],
    )
    evidence = case.get("evidence", {})

    if (
        not isinstance(required, list)
        or not isinstance(evidence, dict)
    ):
        return ["INVALID_EVIDENCE_STATE"]

    return [
        name
        for name in required
        if (
            not isinstance(evidence.get(name), dict)
            or evidence[name].get("received")
            is not True
            or evidence[name].get("verified")
            is not True
        )
    ]


def extract_payload(result: Any) -> dict:
    structured = getattr(
        result,
        "structured_content",
        None,
    )

    if isinstance(structured, dict):
        payload = structured.get(
            "result",
            structured,
        )

        if isinstance(payload, dict):
            return payload

    for block in result.content:
        if not isinstance(block, TextContent):
            continue

        try:
            candidate = json.loads(block.text)
        except json.JSONDecodeError:
            continue

        if isinstance(candidate, dict):
            payload = candidate.get(
                "result",
                candidate,
            )

            if isinstance(payload, dict):
                return payload

    raise RuntimeError(
        "MCP result did not contain a JSON object"
    )


def fetch_dicts(cursor) -> list[dict]:
    columns = [
        str(description[0]).lower()
        for description in cursor.description
    ]

    return [
        dict(zip(columns, row))
        for row in cursor.fetchall()
    ]


required_files = (
    DATA_FILE,
    ORIGINAL_BACKUP_FILE,
    PRE_HUMAN_BACKUP_FILE,
    STATE_TRANSITION_FILE,
    QUALIFICATION_FILE,
    PREFLIGHT_FILE,
    PHASE_B_FILE,
    HUMAN_FILE,
    DATABRICKS_FILE,
    LAB11_FILE,
)

missing_files = [
    str(path)
    for path in required_files
    if not path.is_file()
]

if missing_files:
    raise RuntimeError(
        "Required capstone files are missing: "
        + ", ".join(missing_files)
    )

current_cases = load_json(DATA_FILE)
original_cases = load_json(
    ORIGINAL_BACKUP_FILE
)
pre_human_cases = load_json(
    PRE_HUMAN_BACKUP_FILE
)

state_transition = load_json(
    STATE_TRANSITION_FILE
)
qualification_result = load_json(
    QUALIFICATION_FILE
)
preflight_result = load_json(
    PREFLIGHT_FILE
)
phase_b = load_json(PHASE_B_FILE)
human_result = load_json(HUMAN_FILE)
databricks_result = load_json(
    DATABRICKS_FILE
)
lab11_result = load_json(LAB11_FILE)

current_case = current_cases.get(CASE_ID, {})
original_case = original_cases.get(CASE_ID, {})
pre_human_case = pre_human_cases.get(
    CASE_ID,
    {},
)

original_outstanding = outstanding_evidence(
    original_case
)
pre_human_outstanding = outstanding_evidence(
    pre_human_case
)
current_outstanding = outstanding_evidence(
    current_case
)

human_decision = current_case.get(
    "human_decision",
    {},
)
loss_records = current_case.get(
    "loss_history_records",
    [],
)
wind_evidence = current_case.get(
    "evidence",
    {},
).get(
    "wind_mitigation",
    {},
)

loss_total = (
    sum(
        int(record.get("incurred_loss_usd", 0))
        for record in loss_records
    )
    if isinstance(loss_records, list)
    else -1
)

wind_values = {
    str(value).upper()
    for value in wind_evidence.values()
    if isinstance(
        value,
        (str, int, float, bool),
    )
}

state_history_valid = all(
    (
        set(original_outstanding)
        == {
            "wind_mitigation",
            "contractor_loss_history",
        },
        int(
            original_case.get(
                "state_version",
                0,
            )
        )
        == 0,
        not original_case.get(
            "human_decision"
        ),
        pre_human_outstanding == [],
        pre_human_case.get("state_version")
        == 1,
        not pre_human_case.get(
            "human_decision"
        ),
        current_outstanding == [],
        current_case.get("state_version")
        == 2,
        isinstance(human_decision, dict),
        human_decision.get("decision")
        == "REFER",
        current_case.get("workflow_status")
        == "REFERRED_FOR_SPECIALIST_REVIEW",
        current_case.get(
            "coverage_decision_status"
        )
        == "NOT_MADE",
        isinstance(loss_records, list),
        len(loss_records) == 2,
        loss_total == 325000,
        "PARTIAL" in wind_values,
    )
)

phase_a_record = state_transition.get(
    "phase_a",
    {},
)
transition_record = state_transition.get(
    "state_transition",
    {},
)

phase_a_history_valid = all(
    (
        isinstance(phase_a_record, dict),
        isinstance(transition_record, dict),
        phase_a_record.get("route")
        == "NO_LLM",
        phase_a_record.get(
            "control_plane_status"
        )
        == "COMPLETED",
        phase_a_record.get("model_id")
        == "NO_LLM",
        phase_a_record.get("model_calls")
        == 0,
        float(
            phase_a_record.get(
                "estimated_model_cost_usd",
                -1,
            )
        )
        == 0.0,
        phase_a_record.get("response_text")
        == "REQUEST_EVIDENCE",
        phase_a_record.get(
            "authorization_status"
        )
        == "PERMITTED",
        phase_a_record.get(
            "external_action_executed"
        )
        is False,
        phase_a_record.get(
            "coverage_decision_executed"
        )
        is False,
        transition_record.get(
            "state_changed"
        )
        is True,
        state_transition.get(
            "local_state_valid"
        )
        is True,
        state_transition.get(
            "separate_process_valid"
        )
        is True,
        state_transition.get(
            "transition_verified"
        )
        is True,
        int(
            state_transition.get(
                "model_calls",
                -1,
            )
        )
        == 0,
        float(
            state_transition.get(
                "estimated_model_cost_usd",
                -1,
            )
        )
        == 0.0,
        int(
            state_transition.get(
                "external_actions_executed",
                -1,
            )
        )
        == 0,
    )
)

qualification_history_valid = any_true(
    qualification_result,
    "qualification_verified",
    "training_qualification_verified",
)

preflight_history_valid = any_true(
    preflight_result,
    "preflight_verified",
)

phase_b_validation = phase_b.get(
    "validation",
    {},
)
phase_b_metrics = phase_b.get(
    "metrics",
    {},
)
phase_b_selection = phase_b.get(
    "model_selection",
    {},
)

phase_b_valid = all(
    (
        phase_b.get("recommendation") == "REFER",
        phase_b.get("authorization_status")
        == "PENDING_HUMAN_DECISION",
        phase_b_validation.get(
            "phase_b_verified"
        )
        is True,
        phase_b_selection.get("model_id")
        == MODEL_ID,
        phase_b_metrics.get(
            "claude_api_calls"
        )
        == 3,
        phase_b_metrics.get(
            "mcp_tool_calls"
        )
        == 5,
        phase_b_metrics.get("input_tokens")
        == 5560,
        phase_b_metrics.get("output_tokens")
        == 662,
        math.isclose(
            float(
                phase_b_metrics.get(
                    "actual_model_cost_usd",
                    -1,
                )
            ),
            0.00887,
            rel_tol=0.0,
            abs_tol=0.000000001,
        ),
        phase_b_metrics.get(
            "actual_model_cost_usd"
        )
        <= phase_b_metrics.get(
            "cost_limit_usd"
        ),
        phase_b.get(
            "external_action_executed"
        )
        is False,
        phase_b.get(
            "coverage_decision_executed"
        )
        is False,
    )
)

human_decision_valid = all(
    (
        human_result.get(
            "human_decision_verified"
        )
        is True,
        human_decision.get("decision_id")
        == DECISION_ID,
        human_decision.get("decision")
        == "REFER",
        human_decision.get(
            "ai_recommendation"
        )
        == "REFER",
        human_decision.get(
            "ai_recommendation_followed"
        )
        is True,
        human_decision.get(
            "human_confirmation"
        )
        is True,
        human_decision.get("source")
        == "EXPLICIT_USER_INPUT",
        human_decision.get(
            "authorization_status"
        )
        == "HUMAN_AUTHORIZED_DECISION",
    )
)

persistence_artifact_valid = all(
    (
        databricks_result.get(
            "persistence_verified"
        )
        is True,
        databricks_result.get(
            "decision_audit_verified"
        )
        is True,
        databricks_result.get(
            "usage_ledger_verified"
        )
        is True,
        databricks_result.get(
            "external_action_executed"
        )
        is False,
        databricks_result.get(
            "coverage_decision_executed"
        )
        is False,
    )
)

lab11_report = lab11_result.get(
    "report",
    {},
)

lab11_valid = all(
    (
        isinstance(lab11_report, dict),
        lab11_report.get("lab_verified")
        is True,
        lab11_report.get(
            "happy_path_valid"
        )
        is True,
        lab11_report.get(
            "all_results_safe"
        )
        is True,
        lab11_report.get(
            "telemetry_valid"
        )
        is True,
        lab11_report.get(
            "human_authorization_enforced"
        )
        is True,
        int(
            lab11_report.get(
                "claude_api_calls",
                -1,
            )
        )
        == 0,
        int(
            lab11_report.get(
                "databricks_writes",
                -1,
            )
        )
        == 0,
        int(
            lab11_report.get(
                "underwriting_actions_executed",
                -1,
            )
        )
        == 0,
    )
)


async def main() -> None:
    async with Client(
        mcp,
        raise_exceptions=True,
    ) as mcp_client:
        listed = await mcp_client.list_tools()

        server_tools = {
            tool.name
            for tool in listed.tools
        }

        submission_result = (
            await mcp_client.call_tool(
                "submission",
                {"case_id": CASE_ID},
            )
        )
        compliance_result = (
            await mcp_client.call_tool(
                "compliance_rules",
                {"case_id": CASE_ID},
            )
        )
        loss_result = (
            await mcp_client.call_tool(
                "loss_history",
                {"case_id": CASE_ID},
            )
        )

        if any(
            result.is_error
            for result in (
                submission_result,
                compliance_result,
                loss_result,
            )
        ):
            raise RuntimeError(
                "One or more MCP audit calls failed"
            )

        submission_payload = extract_payload(
            submission_result
        )
        compliance_payload = extract_payload(
            compliance_result
        )
        loss_payload = extract_payload(
            loss_result
        )

    mcp_valid = all(
        (
            server_tools == EXPECTED_TOOLS,
            submission_payload.get(
                "evidence_status"
            )
            == "COMPLETE",
            submission_payload.get(
                "outstanding_evidence"
            )
            == [],
            compliance_payload.get(
                "next_step"
            )
            == "HUMAN_REVIEW",
            compliance_payload.get(
                "human_review_required"
            )
            is True,
            compliance_payload.get(
                "human_approval_required"
            )
            is True,
            loss_payload.get(
                "evidence_status"
            )
            == "VERIFIED",
            loss_payload.get("received")
            is True,
            loss_payload.get("verified")
            is True,
            isinstance(
                loss_payload.get("records"),
                list,
            ),
            len(loss_payload.get("records"))
            == 2,
        )
    )

    hostname = os.getenv(
        "DATABRICKS_SERVER_HOSTNAME",
        "",
    ).strip()
    http_path = os.getenv(
        "DATABRICKS_HTTP_PATH",
        "",
    ).strip()
    auth_type = os.getenv(
        "DATABRICKS_AUTH_TYPE",
        "",
    ).strip()

    configuration_valid = all(
        (
            bool(hostname),
            "://" not in hostname,
            http_path.startswith(
                "/sql/1.0/warehouses/"
            ),
            auth_type == "databricks-oauth",
        )
    )

    if not configuration_valid:
        raise RuntimeError(
            "Databricks configuration is invalid"
        )

    print("OPENING DATABRICKS AUDIT CONNECTION...")

    with sql.connect(
        server_hostname=hostname,
        http_path=http_path,
        auth_type=auth_type,
    ) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    catalog_entry_id,
                    route_name,
                    model_id,
                    governance_status,
                    active,
                    allowed_for_consequential_decisions,
                    qualification_sample_size
                FROM workspace.ai_control_plane
                    .model_catalog
                WHERE catalog_entry_id = :entry_id
                """,
                parameters={
                    "entry_id": (
                        "haiku_45_premium_capstone"
                    )
                },
            )
            catalog_rows = fetch_dicts(cursor)

            cursor.execute(
                """
                SELECT
                    decision_id,
                    case_id,
                    state_version,
                    evidence_status,
                    policy_next_step,
                    ai_recommendation,
                    human_decision,
                    human_reason,
                    ai_recommendation_followed,
                    human_confirmation,
                    decision_scope,
                    authorization_status,
                    workflow_status,
                    coverage_decision_status,
                    model_id,
                    model_calls,
                    input_tokens,
                    output_tokens,
                    estimated_model_cost_usd,
                    external_action_executed,
                    coverage_decision_executed,
                    decision_time,
                    recorded_at
                FROM workspace.ai_control_plane
                    .underwriting_decision_audit
                WHERE decision_id = :decision_id
                """,
                parameters={
                    "decision_id": DECISION_ID
                },
            )
            decision_rows = fetch_dicts(cursor)

            cursor.execute(
                """
                SELECT
                    event_id,
                    source_lab,
                    route_name,
                    model_id,
                    model_calls,
                    input_tokens,
                    output_tokens,
                    estimated_model_cost_usd,
                    latency_ms,
                    outcome_valid,
                    external_action_executed,
                    event_time
                FROM workspace.ai_control_plane
                    .model_usage_ledger
                WHERE event_id = :event_id
                """,
                parameters={
                    "event_id": LEDGER_EVENT_ID
                },
            )
            ledger_rows = fetch_dicts(cursor)

    catalog_row = (
        catalog_rows[0]
        if len(catalog_rows) == 1
        else {}
    )
    decision_row = (
        decision_rows[0]
        if len(decision_rows) == 1
        else {}
    )
    ledger_row = (
        ledger_rows[0]
        if len(ledger_rows) == 1
        else {}
    )

    catalog_valid = all(
        (
            len(catalog_rows) == 1,
            catalog_row.get("route_name")
            == "PREMIUM_REASONING",
            catalog_row.get("model_id")
            == MODEL_ID,
            catalog_row.get(
                "governance_status"
            )
            == "QUALIFIED_FOR_TRAINING",
            catalog_row.get("active") is True,
            catalog_row.get(
                "allowed_for_"
                "consequential_decisions"
            )
            is False,
            int(
                catalog_row.get(
                    "qualification_sample_size",
                    0,
                )
            )
            >= 2,
        )
    )

    decision_audit_valid = all(
        (
            len(decision_rows) == 1,
            decision_row.get("case_id")
            == CASE_ID,
            int(
                decision_row.get(
                    "state_version",
                    -1,
                )
            )
            == 2,
            decision_row.get(
                "evidence_status"
            )
            == "COMPLETE",
            decision_row.get(
                "policy_next_step"
            )
            == "HUMAN_REVIEW",
            decision_row.get(
                "ai_recommendation"
            )
            == "REFER",
            decision_row.get(
                "human_decision"
            )
            == "REFER",
            decision_row.get(
                "human_reason"
            )
            == human_decision.get("reason"),
            decision_row.get(
                "ai_recommendation_followed"
            )
            is True,
            decision_row.get(
                "human_confirmation"
            )
            is True,
            decision_row.get(
                "decision_scope"
            )
            == "SYNTHETIC_TRAINING_ONLY",
            decision_row.get(
                "authorization_status"
            )
            == "HUMAN_AUTHORIZED_DECISION",
            decision_row.get(
                "workflow_status"
            )
            == "REFERRED_FOR_SPECIALIST_REVIEW",
            decision_row.get(
                "coverage_decision_status"
            )
            == "NOT_MADE",
            decision_row.get("model_id")
            == MODEL_ID,
            int(
                decision_row.get(
                    "model_calls",
                    -1,
                )
            )
            == 3,
            math.isclose(
                float(
                    decision_row.get(
                        "estimated_model_cost_usd",
                        -1,
                    )
                ),
                0.00887,
                rel_tol=0.0,
                abs_tol=0.000000001,
            ),
            decision_row.get(
                "external_action_executed"
            )
            is False,
            decision_row.get(
                "coverage_decision_executed"
            )
            is False,
            decision_row.get("decision_time")
            is not None,
            decision_row.get("recorded_at")
            is not None,
        )
    )

    ledger_valid = all(
        (
            len(ledger_rows) == 1,
            ledger_row.get("source_lab")
            == "V2_CAPSTONE_PHASE_B",
            ledger_row.get("route_name")
            == "PREMIUM_REASONING",
            ledger_row.get("model_id")
            == MODEL_ID,
            int(
                ledger_row.get(
                    "model_calls",
                    -1,
                )
            )
            == 3,
            int(
                ledger_row.get(
                    "input_tokens",
                    -1,
                )
            )
            == 5560,
            int(
                ledger_row.get(
                    "output_tokens",
                    -1,
                )
            )
            == 662,
            math.isclose(
                float(
                    ledger_row.get(
                        "estimated_model_cost_usd",
                        -1,
                    )
                ),
                0.00887,
                rel_tol=0.0,
                abs_tol=0.000000001,
            ),
            ledger_row.get("outcome_valid")
            is True,
            ledger_row.get(
                "external_action_executed"
            )
            is False,
            ledger_row.get("event_time")
            is not None,
        )
    )

    human_events = [
        event
        for event in load_events()
        if event.get("event_id")
        == "CAPSTONE-CASE001-HUMAN-DECISION-001"
    ]

    human_telemetry_valid = all(
        (
            len(human_events) == 1,
            human_events[0].get(
                "human_decision"
            )
            == "REFER",
            human_events[0].get(
                "external_action_executed"
            )
            is False,
            human_events[0].get(
                "coverage_decision_executed"
            )
            is False,
        )
    )

    authorization_boundary_valid = all(
        (
            current_case.get(
                "external_action_executed"
            )
            is False,
            current_case.get(
                "coverage_decision_executed"
            )
            is False,
            current_case.get(
                "coverage_decision_status"
            )
            == "NOT_MADE",
            phase_b.get(
                "external_action_executed"
            )
            is False,
            phase_b.get(
                "coverage_decision_executed"
            )
            is False,
            decision_row.get(
                "external_action_executed"
            )
            is False,
            decision_row.get(
                "coverage_decision_executed"
            )
            is False,
            catalog_row.get(
                "allowed_for_"
                "consequential_decisions"
            )
            is False,
        )
    )

    stage_checks = {
        "lab11_control_plane": lab11_valid,
        "phase_a_and_state_transition": (
            phase_a_history_valid
            and state_history_valid
        ),
        "scoped_model_qualification": (
            qualification_history_valid
            and catalog_valid
        ),
        "phase_b_preflight": (
            preflight_history_valid
        ),
        "phase_b_recommendation": (
            phase_b_valid
        ),
        "governed_mcp_shared_state": (
            mcp_valid
        ),
        "human_decision": (
            human_decision_valid
        ),
        "local_human_telemetry": (
            human_telemetry_valid
        ),
        "databricks_persistence_artifact": (
            persistence_artifact_valid
        ),
        "databricks_decision_audit": (
            decision_audit_valid
        ),
        "databricks_usage_ledger": (
            ledger_valid
        ),
        "authorization_boundary": (
            authorization_boundary_valid
        ),
    }

    audit_verified_before_event = all(
        stage_checks.values()
    )

    if (
        audit_verified_before_event
        and not any(
            event.get("event_id")
            == FINAL_EVENT_ID
            for event in load_events()
        )
    ):
        record_event(
            {
                "event_id": FINAL_EVENT_ID,
                "lab": "V2_CAPSTONE",
                "event_type": (
                    "FINAL_CAPSTONE_AUDIT"
                ),
                "case_id": CASE_ID,
                "decision_id": DECISION_ID,
                "route": "AUDIT_ONLY",
                "model_id": "NO_MODEL_CALL",
                "model_calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "estimated_model_cost_usd": 0.0,
                "human_decision": "REFER",
                "workflow_status": (
                    "REFERRED_FOR_SPECIALIST_REVIEW"
                ),
                "external_action_executed": False,
                "coverage_decision_executed": False,
                "outcome_valid": True,
                "event_time": datetime.now(
                    timezone.utc
                ).isoformat(),
            }
        )

    final_events = [
        event
        for event in load_events()
        if event.get("event_id")
        == FINAL_EVENT_ID
    ]

    final_event_valid = (
        len(final_events) == 1
        and final_events[0].get(
            "outcome_valid"
        )
        is True
        and final_events[0].get(
            "model_calls"
        )
        == 0
        and final_events[0].get(
            "external_action_executed"
        )
        is False
        and final_events[0].get(
            "coverage_decision_executed"
        )
        is False
    )

    final_verified = all(
        (
            audit_verified_before_event,
            final_event_valid,
        )
    )

    report = {
        "case_id": CASE_ID,
        "decision_id": DECISION_ID,
        "final_status": (
            "COMPLETE"
            if final_verified
            else "FAILED_VALIDATION"
        ),
        "state_history": {
            "initial_state_version": int(
                original_case.get(
                    "state_version",
                    0,
                )
            ),
            "evidence_complete_state_version": (
                pre_human_case.get(
                    "state_version"
                )
            ),
            "human_decision_state_version": (
                current_case.get(
                    "state_version"
                )
            ),
            "initial_policy_next_step": (
                "REQUEST_EVIDENCE"
            ),
            "current_policy_next_step": (
                "HUMAN_REVIEW"
            ),
            "current_workflow_status": (
                current_case.get(
                    "workflow_status"
                )
            ),
        },
        "underwriting_result": {
            "evidence_status": "COMPLETE",
            "wind_mitigation": "PARTIAL",
            "verified_loss_records": len(
                loss_records
            ),
            "total_incurred_loss_usd": (
                loss_total
            ),
            "ai_recommendation": "REFER",
            "human_decision": "REFER",
            "coverage_decision_status": (
                "NOT_MADE"
            ),
        },
        "model_economics": {
            "model_id": MODEL_ID,
            "model_calls": 3,
            "input_tokens": 5560,
            "output_tokens": 662,
            "actual_model_cost_usd": 0.00887,
            "cost_limit_usd": 0.01,
            "within_cost_limit": True,
        },
        "mcp_tool_calls_in_final_audit": 3,
        "claude_calls_in_final_audit": 0,
        "databricks_reads_in_final_audit": 3,
        "databricks_writes_in_final_audit": 0,
        "stage_checks": stage_checks,
        "final_event_valid": final_event_valid,
        "external_action_executed": False,
        "coverage_decision_executed": False,
        "capstone_verified": final_verified,
    }

    write_json_atomic(
        FINAL_REPORT_FILE,
        report,
    )

    print("\nCAPSTONE STATE HISTORY:")
    print(
        json.dumps(
            report["state_history"],
            indent=2,
        )
    )

    print("\nCAPSTONE STAGE CHECKS:")
    print(
        json.dumps(
            stage_checks,
            indent=2,
        )
    )

    print("\nCAPSTONE ECONOMICS:")
    print(
        json.dumps(
            report["model_economics"],
            indent=2,
        )
    )

    print(
        "\nFINAL CAPSTONE AUDIT VERIFIED:",
        final_verified,
    )
    print("MCP TOOL CALLS: 3")
    print("CLAUDE API CALLS: 0")
    print("DATABRICKS READS: 3")
    print("DATABRICKS WRITES: 0")
    print("EXTERNAL ACTIONS EXECUTED: 0")
    print("COVERAGE DECISIONS EXECUTED: 0")
    print(
        "FINAL REPORT FILE:",
        FINAL_REPORT_FILE.name,
    )
    print("SECRETS PRINTED: False")

    if not final_verified:
        failed = [
            name
            for name, value in stage_checks.items()
            if not value
        ]

        raise RuntimeError(
            "Final capstone validation failed: "
            + ", ".join(failed)
        )

    print("V2 CAPSTONE STATUS: COMPLETE")


if __name__ == "__main__":
    anyio.run(main)
