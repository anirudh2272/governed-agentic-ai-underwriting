import json
import math
import os
from pathlib import Path

from databricks import sql
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "underwriting_cases.json"
)
HUMAN_FILE = (
    PROJECT_ROOT
    / "v2"
    / "data"
    / "capstone_human_decision.json"
)
PHASE_B_FILE = (
    PROJECT_ROOT
    / "v2"
    / "data"
    / "capstone_phase_b_recommendation.json"
)
RESULT_FILE = (
    PROJECT_ROOT
    / "v2"
    / "data"
    / "capstone_databricks_persistence.json"
)

AUDIT_TABLE = (
    "workspace.ai_control_plane."
    "underwriting_decision_audit"
)
LEDGER_TABLE = (
    "workspace.ai_control_plane."
    "model_usage_ledger"
)

CASE_ID = "CASE-001"
DECISION_ID = "CAPSTONE-CASE001-DECISION-001"
LEDGER_EVENT_ID = "CAPSTONE-CASE001-PHASEB-001"

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


def fetch_dicts(cursor) -> list[dict]:
    columns = [
        str(description[0]).lower()
        for description in cursor.description
    ]

    return [
        dict(zip(columns, row))
        for row in cursor.fetchall()
    ]


def values_match(
    actual: dict,
    expected: dict,
) -> bool:
    for key, expected_value in expected.items():
        actual_value = actual.get(key)

        if isinstance(expected_value, bool):
            if actual_value is None:
                return False

            if bool(actual_value) != expected_value:
                return False

        elif isinstance(expected_value, float):
            try:
                matches = math.isclose(
                    float(actual_value),
                    expected_value,
                    rel_tol=0.0,
                    abs_tol=0.000000001,
                )
            except (TypeError, ValueError):
                return False

            if not matches:
                return False

        elif isinstance(expected_value, int):
            try:
                if int(actual_value) != expected_value:
                    return False
            except (TypeError, ValueError):
                return False

        elif actual_value != expected_value:
            return False

    return True


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


cases = load_json(DATA_FILE)
human_result = load_json(HUMAN_FILE)
phase_b = load_json(PHASE_B_FILE)

case = cases.get(CASE_ID)
decision = (
    case.get("human_decision", {})
    if isinstance(case, dict)
    else {}
)
metrics = phase_b.get("metrics", {})
selection = phase_b.get("model_selection", {})
route_decision = phase_b.get(
    "route_decision",
    {},
)
validation = phase_b.get("validation", {})

local_sources_valid = all(
    (
        isinstance(case, dict),
        case.get("state_version") == 2,
        decision.get("decision_id")
        == DECISION_ID,
        decision.get("decision") == "REFER",
        decision.get("human_confirmation") is True,
        decision.get("source")
        == "EXPLICIT_USER_INPUT",
        decision.get("decision_scope")
        == "SYNTHETIC_TRAINING_ONLY",
        decision.get("authorization_status")
        == "HUMAN_AUTHORIZED_DECISION",
        case.get("workflow_status")
        == "REFERRED_FOR_SPECIALIST_REVIEW",
        case.get("coverage_decision_status")
        == "NOT_MADE",
        case.get("external_action_executed")
        is False,
        case.get("coverage_decision_executed")
        is False,
        human_result.get(
            "human_decision_verified"
        )
        is True,
        phase_b.get("recommendation") == "REFER",
        phase_b.get("authorization_status")
        == "PENDING_HUMAN_DECISION",
        validation.get("phase_b_verified")
        is True,
        route_decision.get("route")
        == "PREMIUM_REASONING",
        selection.get("selection_status")
        == "SELECTED",
        isinstance(selection.get("model_id"), str),
        metrics.get("claude_api_calls") == 3,
        metrics.get("input_tokens") == 5560,
        metrics.get("output_tokens") == 662,
        math.isclose(
            float(
                metrics.get(
                    "actual_model_cost_usd",
                    -1,
                )
            ),
            0.00887,
            rel_tol=0.0,
            abs_tol=0.000000001,
        ),
        metrics.get("actual_model_cost_usd")
        <= metrics.get("cost_limit_usd"),
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

if not local_sources_valid:
    raise RuntimeError(
        "Local audit-source validation failed"
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

expected_decision = {
    "decision_id": DECISION_ID,
    "case_id": CASE_ID,
    "state_version": 2,
    "evidence_status": "COMPLETE",
    "policy_id": decision.get("policy_id"),
    "policy_next_step": "HUMAN_REVIEW",
    "ai_recommendation": "REFER",
    "human_decision": "REFER",
    "human_reason": decision.get("reason"),
    "ai_recommendation_followed": True,
    "human_actor": decision.get("human_actor"),
    "human_confirmation": True,
    "decision_scope": "SYNTHETIC_TRAINING_ONLY",
    "authorization_status": (
        "HUMAN_AUTHORIZED_DECISION"
    ),
    "workflow_status": (
        "REFERRED_FOR_SPECIALIST_REVIEW"
    ),
    "coverage_decision_status": "NOT_MADE",
    "model_id": selection.get("model_id"),
    "model_calls": int(
        metrics.get("claude_api_calls")
    ),
    "input_tokens": int(
        metrics.get("input_tokens")
    ),
    "output_tokens": int(
        metrics.get("output_tokens")
    ),
    "estimated_model_cost_usd": float(
        metrics.get("actual_model_cost_usd")
    ),
    "external_action_executed": False,
    "coverage_decision_executed": False,
}

decision_parameters = {
    **expected_decision,
    "decision_time": decision.get("decided_at"),
}

expected_ledger = {
    "event_id": LEDGER_EVENT_ID,
    "source_lab": "V2_CAPSTONE_PHASE_B",
    "route_name": "PREMIUM_REASONING",
    "model_id": selection.get("model_id"),
    "model_calls": int(
        metrics.get("claude_api_calls")
    ),
    "input_tokens": int(
        metrics.get("input_tokens")
    ),
    "output_tokens": int(
        metrics.get("output_tokens")
    ),
    "estimated_model_cost_usd": float(
        metrics.get("actual_model_cost_usd")
    ),
    "latency_ms": float(
        metrics.get("model_latency_ms")
    ),
    "outcome_valid": True,
    "external_action_executed": False,
}

audit_columns = {
    "decision_id",
    "case_id",
    "state_version",
    "evidence_status",
    "policy_id",
    "policy_next_step",
    "ai_recommendation",
    "human_decision",
    "human_reason",
    "ai_recommendation_followed",
    "human_actor",
    "human_confirmation",
    "decision_scope",
    "authorization_status",
    "workflow_status",
    "coverage_decision_status",
    "model_id",
    "model_calls",
    "input_tokens",
    "output_tokens",
    "estimated_model_cost_usd",
    "external_action_executed",
    "coverage_decision_executed",
    "decision_time",
    "recorded_at",
}

create_audit_table = f"""
CREATE TABLE IF NOT EXISTS {AUDIT_TABLE} (
    decision_id STRING,
    case_id STRING,
    state_version BIGINT,
    evidence_status STRING,
    policy_id STRING,
    policy_next_step STRING,
    ai_recommendation STRING,
    human_decision STRING,
    human_reason STRING,
    ai_recommendation_followed BOOLEAN,
    human_actor STRING,
    human_confirmation BOOLEAN,
    decision_scope STRING,
    authorization_status STRING,
    workflow_status STRING,
    coverage_decision_status STRING,
    model_id STRING,
    model_calls BIGINT,
    input_tokens BIGINT,
    output_tokens BIGINT,
    estimated_model_cost_usd DOUBLE,
    external_action_executed BOOLEAN,
    coverage_decision_executed BOOLEAN,
    decision_time TIMESTAMP,
    recorded_at TIMESTAMP
)
USING DELTA
COMMENT 'Immutable synthetic underwriting decision audit'
"""

decision_select = f"""
SELECT
    decision_id,
    case_id,
    state_version,
    evidence_status,
    policy_id,
    policy_next_step,
    ai_recommendation,
    human_decision,
    human_reason,
    ai_recommendation_followed,
    human_actor,
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
FROM {AUDIT_TABLE}
WHERE decision_id = :decision_id
"""

ledger_select = f"""
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
FROM {LEDGER_TABLE}
WHERE event_id = :event_id
"""

decision_merge = f"""
MERGE INTO {AUDIT_TABLE} AS target
USING (
    SELECT
        :decision_id AS decision_id,
        :case_id AS case_id,
        CAST(:state_version AS BIGINT)
            AS state_version,
        :evidence_status AS evidence_status,
        :policy_id AS policy_id,
        :policy_next_step AS policy_next_step,
        :ai_recommendation AS ai_recommendation,
        :human_decision AS human_decision,
        :human_reason AS human_reason,
        CAST(:ai_recommendation_followed AS BOOLEAN)
            AS ai_recommendation_followed,
        :human_actor AS human_actor,
        CAST(:human_confirmation AS BOOLEAN)
            AS human_confirmation,
        :decision_scope AS decision_scope,
        :authorization_status
            AS authorization_status,
        :workflow_status AS workflow_status,
        :coverage_decision_status
            AS coverage_decision_status,
        :model_id AS model_id,
        CAST(:model_calls AS BIGINT)
            AS model_calls,
        CAST(:input_tokens AS BIGINT)
            AS input_tokens,
        CAST(:output_tokens AS BIGINT)
            AS output_tokens,
        CAST(:estimated_model_cost_usd AS DOUBLE)
            AS estimated_model_cost_usd,
        CAST(:external_action_executed AS BOOLEAN)
            AS external_action_executed,
        CAST(:coverage_decision_executed AS BOOLEAN)
            AS coverage_decision_executed,
        CAST(:decision_time AS TIMESTAMP)
            AS decision_time
) AS incoming
ON target.decision_id = incoming.decision_id
WHEN NOT MATCHED THEN INSERT (
    decision_id,
    case_id,
    state_version,
    evidence_status,
    policy_id,
    policy_next_step,
    ai_recommendation,
    human_decision,
    human_reason,
    ai_recommendation_followed,
    human_actor,
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
)
VALUES (
    incoming.decision_id,
    incoming.case_id,
    incoming.state_version,
    incoming.evidence_status,
    incoming.policy_id,
    incoming.policy_next_step,
    incoming.ai_recommendation,
    incoming.human_decision,
    incoming.human_reason,
    incoming.ai_recommendation_followed,
    incoming.human_actor,
    incoming.human_confirmation,
    incoming.decision_scope,
    incoming.authorization_status,
    incoming.workflow_status,
    incoming.coverage_decision_status,
    incoming.model_id,
    incoming.model_calls,
    incoming.input_tokens,
    incoming.output_tokens,
    incoming.estimated_model_cost_usd,
    incoming.external_action_executed,
    incoming.coverage_decision_executed,
    incoming.decision_time,
    current_timestamp()
)
"""

ledger_merge = f"""
MERGE INTO {LEDGER_TABLE} AS target
USING (
    SELECT
        :event_id AS event_id,
        :source_lab AS source_lab,
        :route_name AS route_name,
        :model_id AS model_id,
        CAST(:model_calls AS BIGINT)
            AS model_calls,
        CAST(:input_tokens AS BIGINT)
            AS input_tokens,
        CAST(:output_tokens AS BIGINT)
            AS output_tokens,
        CAST(:estimated_model_cost_usd AS DOUBLE)
            AS estimated_model_cost_usd,
        CAST(:latency_ms AS DOUBLE)
            AS latency_ms,
        CAST(:outcome_valid AS BOOLEAN)
            AS outcome_valid,
        CAST(:external_action_executed AS BOOLEAN)
            AS external_action_executed
) AS incoming
ON target.event_id = incoming.event_id
WHEN NOT MATCHED THEN INSERT (
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
)
VALUES (
    incoming.event_id,
    incoming.source_lab,
    incoming.route_name,
    incoming.model_id,
    incoming.model_calls,
    incoming.input_tokens,
    incoming.output_tokens,
    incoming.estimated_model_cost_usd,
    incoming.latency_ms,
    incoming.outcome_valid,
    incoming.external_action_executed,
    current_timestamp()
)
"""

print("LOCAL AUDIT SOURCES VALID:", local_sources_valid)
print("OPENING DATABRICKS CONNECTION...")

with sql.connect(
    server_hostname=hostname,
    http_path=http_path,
    auth_type=auth_type,
) as connection:
    with connection.cursor() as cursor:
        cursor.execute(
            "SHOW TABLES IN "
            "workspace.ai_control_plane"
        )
        table_names_before = {
            str(row[1]).lower()
            for row in cursor.fetchall()
        }

        table_created = (
            "underwriting_decision_audit"
            not in table_names_before
        )

        cursor.execute(create_audit_table)

        cursor.execute(
            f"DESCRIBE TABLE {AUDIT_TABLE}"
        )
        actual_audit_columns = {
            str(row[0]).lower()
            for row in cursor.fetchall()
            if row
            and row[0]
            and not str(row[0]).startswith("#")
        }

        audit_schema_valid = (
            audit_columns
            .issubset(actual_audit_columns)
        )

        if not audit_schema_valid:
            raise RuntimeError(
                "Decision audit table schema is invalid"
            )

        cursor.execute(
            decision_select,
            parameters={
                "decision_id": DECISION_ID
            },
        )
        existing_decisions = fetch_dicts(cursor)

        cursor.execute(
            ledger_select,
            parameters={
                "event_id": LEDGER_EVENT_ID
            },
        )
        existing_ledger = fetch_dicts(cursor)

        if len(existing_decisions) > 1:
            raise RuntimeError(
                "Duplicate decision IDs already exist"
            )

        if len(existing_ledger) > 1:
            raise RuntimeError(
                "Duplicate ledger event IDs already exist"
            )

        if (
            existing_decisions
            and not values_match(
                existing_decisions[0],
                expected_decision,
            )
        ):
            raise RuntimeError(
                "Conflicting decision audit row blocked"
            )

        if (
            existing_ledger
            and not values_match(
                existing_ledger[0],
                expected_ledger,
            )
        ):
            raise RuntimeError(
                "Conflicting usage ledger row blocked"
            )

        decision_status = "ALREADY_EXISTS"
        ledger_status = "ALREADY_EXISTS"

        if not existing_decisions:
            cursor.execute(
                decision_merge,
                parameters=decision_parameters,
            )
            decision_status = "INSERTED"

        if not existing_ledger:
            cursor.execute(
                ledger_merge,
                parameters=expected_ledger,
            )
            ledger_status = "INSERTED"

        cursor.execute(
            decision_select,
            parameters={
                "decision_id": DECISION_ID
            },
        )
        stored_decisions = fetch_dicts(cursor)

        cursor.execute(
            ledger_select,
            parameters={
                "event_id": LEDGER_EVENT_ID
            },
        )
        stored_ledger = fetch_dicts(cursor)

stored_decision = (
    stored_decisions[0]
    if len(stored_decisions) == 1
    else {}
)
stored_usage = (
    stored_ledger[0]
    if len(stored_ledger) == 1
    else {}
)

decision_verified = all(
    (
        len(stored_decisions) == 1,
        values_match(
            stored_decision,
            expected_decision,
        ),
        stored_decision.get(
            "decision_time"
        )
        is not None,
        stored_decision.get(
            "recorded_at"
        )
        is not None,
    )
)

ledger_verified = all(
    (
        len(stored_ledger) == 1,
        values_match(
            stored_usage,
            expected_ledger,
        ),
        stored_usage.get("event_time")
        is not None,
    )
)

rows_inserted = sum(
    (
        decision_status == "INSERTED",
        ledger_status == "INSERTED",
    )
)

persistence_verified = all(
    (
        audit_schema_valid,
        decision_verified,
        ledger_verified,
        stored_decision.get(
            "external_action_executed"
        )
        is False,
        stored_decision.get(
            "coverage_decision_executed"
        )
        is False,
        stored_usage.get(
            "external_action_executed"
        )
        is False,
    )
)

report = {
    "case_id": CASE_ID,
    "decision_id": DECISION_ID,
    "ledger_event_id": LEDGER_EVENT_ID,
    "audit_table": AUDIT_TABLE,
    "ledger_table": LEDGER_TABLE,
    "audit_table_created": table_created,
    "audit_schema_valid": audit_schema_valid,
    "decision_row_status": decision_status,
    "ledger_row_status": ledger_status,
    "rows_inserted": rows_inserted,
    "rows_updated": 0,
    "rows_deleted": 0,
    "human_decision": "REFER",
    "workflow_status": (
        "REFERRED_FOR_SPECIALIST_REVIEW"
    ),
    "model_id": selection.get("model_id"),
    "model_calls": metrics.get(
        "claude_api_calls"
    ),
    "input_tokens": metrics.get(
        "input_tokens"
    ),
    "output_tokens": metrics.get(
        "output_tokens"
    ),
    "model_cost_usd": metrics.get(
        "actual_model_cost_usd"
    ),
    "decision_audit_verified": (
        decision_verified
    ),
    "usage_ledger_verified": ledger_verified,
    "external_action_executed": False,
    "coverage_decision_executed": False,
    "persistence_verified": (
        persistence_verified
    ),
}

write_json_atomic(
    RESULT_FILE,
    report,
)

print("\nDATABRICKS PERSISTENCE RESULT:")
print(
    json.dumps(
        {
            "audit_table_created": (
                table_created
            ),
            "decision_row_status": (
                decision_status
            ),
            "ledger_row_status": (
                ledger_status
            ),
            "rows_inserted": rows_inserted,
            "rows_updated": 0,
            "rows_deleted": 0,
        },
        indent=2,
    )
)

print("\nSTORED DECISION SUMMARY:")
print(
    json.dumps(
        {
            "decision_id": stored_decision.get(
                "decision_id"
            ),
            "case_id": stored_decision.get(
                "case_id"
            ),
            "state_version": stored_decision.get(
                "state_version"
            ),
            "human_decision": (
                stored_decision.get(
                    "human_decision"
                )
            ),
            "workflow_status": (
                stored_decision.get(
                    "workflow_status"
                )
            ),
            "authorization_status": (
                stored_decision.get(
                    "authorization_status"
                )
            ),
            "model_id": stored_decision.get(
                "model_id"
            ),
            "estimated_model_cost_usd": (
                stored_decision.get(
                    "estimated_model_cost_usd"
                )
            ),
        },
        indent=2,
        default=str,
    )
)

print(
    "\nDECISION AUDIT VERIFIED:",
    decision_verified,
)
print(
    "PHASE B USAGE LEDGER VERIFIED:",
    ledger_verified,
)
print(
    "DATABRICKS PERSISTENCE VERIFIED:",
    persistence_verified,
)
print(
    "DATABRICKS TABLES CREATED:",
    int(table_created),
)
print("DATABRICKS ROWS INSERTED:", rows_inserted)
print("DATABRICKS ROWS UPDATED: 0")
print("DATABRICKS ROWS DELETED: 0")
print("AUTHORITATIVE STATE CHANGES: 0")
print("MODEL CALLS EXECUTED: 0")
print("EXTERNAL ACTIONS EXECUTED: 0")
print("COVERAGE DECISIONS EXECUTED: 0")
print("RESULT FILE:", RESULT_FILE.name)
print("SECRETS PRINTED: False")

if not persistence_verified:
    raise RuntimeError(
        "Databricks persistence validation failed"
    )

print(
    "CAPSTONE DATABRICKS PERSISTENCE: COMPLETE"
)
