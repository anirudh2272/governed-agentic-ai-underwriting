"""Create a narrowly scoped premium training qualification."""

import json
import os
from pathlib import Path
from typing import Any

from databricks import sql
from dotenv import load_dotenv

from v2.control_plane.model_selector import (
    load_model_catalog,
    select_model,
)
from v2.control_plane.policies import (
    get_route_policy,
)
from v2.services.telemetry import record_event


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULT_FILE = (
    PROJECT_ROOT
    / "v2"
    / "data"
    / "capstone_model_qualification.json"
)

CATALOG_TABLE = (
    "workspace.ai_control_plane.model_catalog"
)
BENCHMARK_TABLE = (
    "workspace.ai_control_plane.model_benchmarks"
)

CANDIDATE_ID = "haiku_45_premium_candidate"
QUALIFIED_ID = "haiku_45_premium_capstone"

BENCHMARK_IDS = {
    "case001_lab08_controlled_mcp",
    "case001_lab09_single",
}


def fetch_dicts(
    cursor: Any,
    query: str,
) -> list[dict[str, Any]]:
    cursor.execute(query)

    columns = [
        item[0]
        for item in cursor.description
    ]

    return [
        dict(zip(columns, row))
        for row in cursor.fetchall()
    ]


load_dotenv(PROJECT_ROOT / ".env")

hostname = (
    os.getenv("DATABRICKS_SERVER_HOSTNAME")
    or ""
).strip()
http_path = (
    os.getenv("DATABRICKS_HTTP_PATH")
    or ""
).strip()
auth_type = (
    os.getenv("DATABRICKS_AUTH_TYPE")
    or ""
).strip()


print("OPENING DATABRICKS GOVERNANCE CONNECTION...")


with sql.connect(
    server_hostname=hostname,
    http_path=http_path,
    auth_type=auth_type,
) as connection:
    with connection.cursor() as cursor:
        candidate_rows = fetch_dicts(
            cursor,
            f"""
            SELECT *
            FROM {CATALOG_TABLE}
            WHERE catalog_entry_id
                = '{CANDIDATE_ID}'
            """,
        )

        benchmark_rows = fetch_dicts(
            cursor,
            f"""
            SELECT *
            FROM {BENCHMARK_TABLE}
            WHERE benchmark_id IN (
                'case001_lab08_controlled_mcp',
                'case001_lab09_single'
            )
            ORDER BY benchmark_id
            """,
        )

        existing_qualified_rows = fetch_dicts(
            cursor,
            f"""
            SELECT *
            FROM {CATALOG_TABLE}
            WHERE catalog_entry_id
                = '{QUALIFIED_ID}'
            """,
        )

        if len(candidate_rows) != 1:
            raise SystemExit(
                "Expected exactly one premium "
                "candidate catalog entry."
            )

        candidate = candidate_rows[0]

        candidate_valid = all(
            (
                candidate.get("route_name")
                == "PREMIUM_REASONING",
                candidate.get(
                    "governance_status"
                )
                == "EVALUATION_REQUIRED",
                candidate.get("active") is False,
                candidate.get(
                    "allowed_for_"
                    "consequential_decisions"
                )
                is False,
            )
        )

        benchmark_ids_found = {
            row.get("benchmark_id")
            for row in benchmark_rows
        }

        benchmark_evidence_valid = all(
            (
                benchmark_ids_found
                == BENCHMARK_IDS,
                len(benchmark_rows) == 2,
                all(
                    row.get("outcome_valid")
                    is True
                    for row in benchmark_rows
                ),
                all(
                    row.get(
                        "external_action_executed"
                    )
                    is False
                    for row in benchmark_rows
                ),
                all(
                    row.get("model_id")
                    == candidate.get("model_id")
                    for row in benchmark_rows
                ),
            )
        )

        if not candidate_valid:
            raise SystemExit(
                "Premium candidate lifecycle state "
                "is not the expected fail-closed state."
            )

        if not benchmark_evidence_valid:
            raise SystemExit(
                "Qualification benchmark evidence "
                "is incomplete or invalid."
            )

        databricks_writes = 0

        if not existing_qualified_rows:
            cursor.execute(
                f"""
                INSERT INTO {CATALOG_TABLE} (
                    catalog_entry_id,
                    route_name,
                    model_id,
                    provider,
                    capability_tier,
                    governance_status,
                    active,
                    allowed_for_consequential_decisions,
                    input_cost_usd_per_million,
                    output_cost_usd_per_million,
                    qualification_sample_size,
                    evidence_source,
                    price_source,
                    price_verified_date,
                    notes,
                    updated_at
                )
                SELECT
                    '{QUALIFIED_ID}',
                    'PREMIUM_REASONING',
                    model_id,
                    provider,
                    'PREMIUM_REASONING',
                    'QUALIFIED_FOR_TRAINING',
                    TRUE,
                    FALSE,
                    input_cost_usd_per_million,
                    output_cost_usd_per_million,
                    2,
                    'V2_LAB_08_AND_09_CASE001',
                    price_source,
                    price_verified_date,
                    'Training-only scope for synthetic CASE-001 capstone; recommendation only; no consequential authority',
                    CURRENT_TIMESTAMP()
                FROM {CATALOG_TABLE}
                WHERE catalog_entry_id
                    = '{CANDIDATE_ID}'
                """
            )
            databricks_writes = 1

        qualified_rows = fetch_dicts(
            cursor,
            f"""
            SELECT *
            FROM {CATALOG_TABLE}
            WHERE catalog_entry_id
                = '{QUALIFIED_ID}'
            """,
        )

        original_candidate_after = fetch_dicts(
            cursor,
            f"""
            SELECT *
            FROM {CATALOG_TABLE}
            WHERE catalog_entry_id
                = '{CANDIDATE_ID}'
            """,
        )


if len(qualified_rows) != 1:
    raise RuntimeError(
        "Expected one scoped premium "
        "qualification entry."
    )


qualified = qualified_rows[0]
original_candidate = original_candidate_after[0]


qualified_entry_valid = all(
    (
        qualified.get("route_name")
        == "PREMIUM_REASONING",
        qualified.get("model_id")
        == candidate.get("model_id"),
        qualified.get("governance_status")
        == "QUALIFIED_FOR_TRAINING",
        qualified.get("active") is True,
        qualified.get(
            "allowed_for_consequential_decisions"
        )
        is False,
        int(
            qualified.get(
                "qualification_sample_size",
                0,
            )
        )
        == 2,
        qualified.get("evidence_source")
        == "V2_LAB_08_AND_09_CASE001",
    )
)


original_candidate_preserved = all(
    (
        original_candidate.get(
            "governance_status"
        )
        == "EVALUATION_REQUIRED",
        original_candidate.get("active")
        is False,
        original_candidate.get(
            "allowed_for_consequential_decisions"
        )
        is False,
    )
)


catalog = load_model_catalog()

selection = select_model(
    "PREMIUM_REASONING",
    catalog,
    expected_input_tokens=2500,
    expected_output_tokens=500,
    consequential_action_requested=False,
)

premium_policy = get_route_policy(
    "PREMIUM_REASONING"
)

selection_valid = all(
    (
        selection.get("selection_status")
        == "SELECTED",
        selection.get("catalog_entry_id")
        == QUALIFIED_ID,
        selection.get("model_id")
        == qualified.get("model_id"),
        selection.get(
            "allowed_for_"
            "consequential_decisions"
        )
        is False,
        float(
            selection.get(
                "estimated_model_cost_usd",
                999,
            )
        )
        <= premium_policy[
            "max_estimated_model_cost_usd"
        ],
    )
)


qualification_verified = all(
    (
        candidate_valid,
        benchmark_evidence_valid,
        qualified_entry_valid,
        original_candidate_preserved,
        selection_valid,
    )
)


report = {
    "candidate_id": CANDIDATE_ID,
    "qualified_entry_id": QUALIFIED_ID,
    "candidate_preserved": (
        original_candidate_preserved
    ),
    "benchmark_ids": sorted(
        benchmark_ids_found
    ),
    "benchmark_sample_size": len(
        benchmark_rows
    ),
    "benchmark_evidence_valid": (
        benchmark_evidence_valid
    ),
    "qualification_scope": (
        "SYNTHETIC_CASE001_CAPSTONE_ONLY"
    ),
    "qualified_for_production": False,
    "qualified_entry_valid": (
        qualified_entry_valid
    ),
    "selection": selection,
    "selection_valid": selection_valid,
    "databricks_writes": databricks_writes,
    "model_calls": 0,
    "external_actions_executed": 0,
    "qualification_verified": (
        qualification_verified
    ),
}


RESULT_FILE.write_text(
    json.dumps(
        report,
        indent=2,
        default=str,
    )
    + "\n",
    encoding="utf-8",
)


record_event(
    {
        "lab": "V2_CAPSTONE",
        "event_type": (
            "TRAINING_MODEL_QUALIFICATION"
        ),
        "catalog_entry_id": QUALIFIED_ID,
        "route": "PREMIUM_REASONING",
        "model_id": qualified.get("model_id"),
        "qualification_scope": (
            "SYNTHETIC_CASE001_CAPSTONE_ONLY"
        ),
        "qualification_sample_size": 2,
        "qualified_for_production": False,
        "allowed_for_consequential_decisions": (
            False
        ),
        "model_calls": 0,
        "external_action_executed": False,
        "verified": qualification_verified,
    }
)


print("\nQUALIFICATION EVIDENCE:")
print(
    json.dumps(
        [
            {
                "benchmark_id": row.get(
                    "benchmark_id"
                ),
                "source_lab": row.get(
                    "source_lab"
                ),
                "model_id": row.get(
                    "model_id"
                ),
                "outcome_valid": row.get(
                    "outcome_valid"
                ),
                "external_action_executed": (
                    row.get(
                        "external_action_executed"
                    )
                ),
            }
            for row in benchmark_rows
        ],
        indent=2,
    )
)


print("\nSCOPED CATALOG ENTRY:")
print(
    json.dumps(
        {
            "catalog_entry_id": (
                qualified.get(
                    "catalog_entry_id"
                )
            ),
            "route_name": qualified.get(
                "route_name"
            ),
            "model_id": qualified.get(
                "model_id"
            ),
            "governance_status": (
                qualified.get(
                    "governance_status"
                )
            ),
            "active": qualified.get("active"),
            "allowed_for_consequential_decisions": (
                qualified.get(
                    "allowed_for_"
                    "consequential_decisions"
                )
            ),
            "qualification_sample_size": (
                qualified.get(
                    "qualification_sample_size"
                )
            ),
            "evidence_source": qualified.get(
                "evidence_source"
            ),
        },
        indent=2,
        default=str,
    )
)


print("\nPREMIUM SELECTION:")
print(json.dumps(selection, indent=2))


print(
    "\nORIGINAL CANDIDATE PRESERVED:",
    original_candidate_preserved,
)
print(
    "BENCHMARK EVIDENCE VALID:",
    benchmark_evidence_valid,
)
print(
    "SCOPED QUALIFICATION VALID:",
    qualified_entry_valid,
)
print(
    "PREMIUM SELECTION VALID:",
    selection_valid,
)
print(
    "TRAINING QUALIFICATION VERIFIED:",
    qualification_verified,
)
print(
    "QUALIFIED FOR PRODUCTION: False"
)
print(
    "CONSEQUENTIAL DECISION AUTHORITY: False"
)
print(
    "DATABRICKS WRITES:",
    databricks_writes,
)
print("MODEL CALLS: 0")
print("UNDERWRITING ACTIONS EXECUTED: 0")
print("SECRETS PRINTED: False")


if not qualification_verified:
    raise RuntimeError(
        "Training qualification failed."
    )


print(
    "CAPSTONE MODEL QUALIFICATION: COMPLETE"
)
