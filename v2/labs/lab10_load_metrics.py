import json
import os
from pathlib import Path
from typing import Any

from databricks import sql
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]

BENCHMARK_TABLE = (
    "workspace.ai_control_plane."
    "model_benchmarks"
)
LEDGER_TABLE = (
    "workspace.ai_control_plane."
    "model_usage_ledger"
)


def read_telemetry() -> list[dict[str, Any]]:
    telemetry_file = (
        PROJECT_ROOT
        / "v2"
        / "data"
        / "interaction_telemetry.jsonl"
    )

    return [
        json.loads(line)
        for line in telemetry_file.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]


def latest_event(
    events: list[dict[str, Any]],
    lab_name: str,
) -> dict[str, Any]:
    matching = [
        event
        for event in events
        if event.get("lab") == lab_name
    ]

    if not matching:
        raise RuntimeError(
            f"No telemetry found for {lab_name}."
        )

    return matching[-1]


def upsert_row(
    cursor,
    table_name: str,
    key_column: str,
    timestamp_column: str,
    row: dict[str, Any],
) -> str:
    key_value = row[key_column]

    cursor.execute(
        f"""
        SELECT COUNT(*)
        FROM {table_name}
        WHERE {key_column} = ?
        """,
        [key_value],
    )
    exists = cursor.fetchone()[0] > 0

    columns = list(row)
    non_key_columns = [
        column
        for column in columns
        if column != key_column
    ]

    if exists:
        assignments = ", ".join(
            f"{column} = ?"
            for column in non_key_columns
        )

        parameters = [
            row[column]
            for column in non_key_columns
        ]
        parameters.append(key_value)

        cursor.execute(
            f"""
            UPDATE {table_name}
            SET
                {assignments},
                {timestamp_column}
                    = current_timestamp()
            WHERE {key_column} = ?
            """,
            parameters,
        )
        return "UPDATED"

    insert_columns = columns + [
        timestamp_column
    ]
    placeholders = ", ".join(
        "?"
        for _ in columns
    )

    cursor.execute(
        f"""
        INSERT INTO {table_name} (
            {", ".join(insert_columns)}
        )
        VALUES (
            {placeholders},
            current_timestamp()
        )
        """,
        [
            row[column]
            for column in columns
        ],
    )
    return "INSERTED"


load_dotenv(PROJECT_ROOT / ".env")

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
configured_model = os.getenv(
    "CLAUDE_MODEL",
    "",
).strip()

if not all(
    (
        hostname,
        http_path,
        configured_model,
    )
):
    raise SystemExit(
        "Required configuration is missing."
    )

if auth_type != "databricks-oauth":
    raise SystemExit(
        "Expected databricks-oauth."
    )

lab9_file = (
    PROJECT_ROOT
    / "v2"
    / "data"
    / "lab09_economics_results.json"
)

if not lab9_file.exists():
    raise SystemExit(
        "Lab 9 economics result is missing."
    )

lab9 = json.loads(
    lab9_file.read_text(encoding="utf-8")
)
events = read_telemetry()
lab8 = latest_event(events, "V2_LAB_08")

if lab8["model_id"] != configured_model:
    raise RuntimeError(
        "Lab 8 model differs from current model."
    )

if lab9["model_id"] != configured_model:
    raise RuntimeError(
        "Lab 9 model differs from current model."
    )

rates = lab9["pricing"]

lab8_cost = (
    lab8["input_tokens"]
    / 1_000_000
    * rates["input_per_million"]
    + lab8["output_tokens"]
    / 1_000_000
    * rates["output_per_million"]
)

single = lab9[
    "single_agent"
]["economics"]
multi = lab9[
    "multi_agent"
]["economics"]

benchmark_rows = [
    {
        "benchmark_id": (
            "case001_lab08_controlled_mcp"
        ),
        "model_id": configured_model,
        "route_name": (
            "CONTROLLED_MCP_AGENT"
        ),
        "orchestration_pattern": (
            "TOOL_USING_SINGLE_AGENT"
        ),
        "source_lab": "V2_LAB_08",
        "case_id": "CASE-001",
        "outcome_valid": bool(
            lab8["lab_verified"]
        ),
        "model_calls": int(
            lab8["model_calls"]
        ),
        "input_tokens": int(
            lab8["input_tokens"]
        ),
        "output_tokens": int(
            lab8["output_tokens"]
        ),
        "latency_ms": float(
            lab8["latency_ms"]
        ),
        "estimated_model_cost_usd": (
            float(lab8_cost)
        ),
        "external_action_executed": (
            bool(
                lab8[
                    "external_action_executed"
                ]
            )
        ),
    },
    {
        "benchmark_id": (
            "case001_lab09_single"
        ),
        "model_id": configured_model,
        "route_name": (
            "PREFETCHED_SINGLE_AGENT"
        ),
        "orchestration_pattern": (
            "SINGLE_AGENT"
        ),
        "source_lab": "V2_LAB_09",
        "case_id": "CASE-001",
        "outcome_valid": bool(
            lab9[
                "single_agent"
            ]["validation"]["valid"]
        ),
        "model_calls": int(
            single["model_calls"]
        ),
        "input_tokens": int(
            single["input_tokens"]
        ),
        "output_tokens": int(
            single["output_tokens"]
        ),
        "latency_ms": float(
            single[
                "sequential_latency_ms"
            ]
        ),
        "estimated_model_cost_usd": (
            float(
                single[
                    "estimated_total_cost_usd"
                ]
            )
        ),
        "external_action_executed": False,
    },
    {
        "benchmark_id": (
            "case001_lab09_multi"
        ),
        "model_id": configured_model,
        "route_name": (
            "PREFETCHED_MULTI_AGENT"
        ),
        "orchestration_pattern": (
            "MULTI_AGENT"
        ),
        "source_lab": "V2_LAB_09",
        "case_id": "CASE-001",
        "outcome_valid": bool(
            lab9[
                "multi_agent"
            ]["team_valid"]
        ),
        "model_calls": int(
            multi["model_calls"]
        ),
        "input_tokens": int(
            multi["input_tokens"]
        ),
        "output_tokens": int(
            multi["output_tokens"]
        ),
        "latency_ms": float(
            multi[
                "sequential_latency_ms"
            ]
        ),
        "estimated_model_cost_usd": (
            float(
                multi[
                    "estimated_total_cost_usd"
                ]
            )
        ),
        "external_action_executed": False,
    },
]

ledger_rows = [
    {
        "event_id": (
            "usage_case001_lab08_controlled"
        ),
        "source_lab": "V2_LAB_08",
        "route_name": (
            "CONTROLLED_MCP_AGENT"
        ),
        "model_id": configured_model,
        "model_calls": int(
            lab8["model_calls"]
        ),
        "input_tokens": int(
            lab8["input_tokens"]
        ),
        "output_tokens": int(
            lab8["output_tokens"]
        ),
        "estimated_model_cost_usd": (
            float(lab8_cost)
        ),
        "latency_ms": float(
            lab8["latency_ms"]
        ),
        "outcome_valid": bool(
            lab8["lab_verified"]
        ),
        "external_action_executed": (
            bool(
                lab8[
                    "external_action_executed"
                ]
            )
        ),
    },
    {
        "event_id": (
            "usage_case001_lab09_single"
        ),
        "source_lab": "V2_LAB_09",
        "route_name": (
            "PREFETCHED_SINGLE_AGENT"
        ),
        "model_id": configured_model,
        "model_calls": int(
            single["model_calls"]
        ),
        "input_tokens": int(
            single["input_tokens"]
        ),
        "output_tokens": int(
            single["output_tokens"]
        ),
        "estimated_model_cost_usd": (
            float(
                single[
                    "estimated_total_cost_usd"
                ]
            )
        ),
        "latency_ms": float(
            single[
                "sequential_latency_ms"
            ]
        ),
        "outcome_valid": bool(
            lab9[
                "single_agent"
            ]["validation"]["valid"]
        ),
        "external_action_executed": False,
    },
    {
        "event_id": (
            "usage_case001_lab09_multi"
        ),
        "source_lab": "V2_LAB_09",
        "route_name": (
            "PREFETCHED_MULTI_AGENT"
        ),
        "model_id": configured_model,
        "model_calls": int(
            multi["model_calls"]
        ),
        "input_tokens": int(
            multi["input_tokens"]
        ),
        "output_tokens": int(
            multi["output_tokens"]
        ),
        "estimated_model_cost_usd": (
            float(
                multi[
                    "estimated_total_cost_usd"
                ]
            )
        ),
        "latency_ms": float(
            multi[
                "sequential_latency_ms"
            ]
        ),
        "outcome_valid": bool(
            lab9[
                "multi_agent"
            ]["team_valid"]
        ),
        "external_action_executed": False,
    },
]

expected_benchmark_ids = {
    row["benchmark_id"]
    for row in benchmark_rows
}
expected_event_ids = {
    row["event_id"]
    for row in ledger_rows
}

inserted = 0
updated = 0

print("OPENING DATABRICKS CONNECTION...")

with sql.connect(
    server_hostname=hostname,
    http_path=http_path,
    auth_type=auth_type,
    use_cloud_fetch=False,
) as connection:
    with connection.cursor() as cursor:
        for row in benchmark_rows:
            status = upsert_row(
                cursor,
                BENCHMARK_TABLE,
                "benchmark_id",
                "measured_at",
                row,
            )

            if status == "INSERTED":
                inserted += 1
            else:
                updated += 1

        for row in ledger_rows:
            status = upsert_row(
                cursor,
                LEDGER_TABLE,
                "event_id",
                "event_time",
                row,
            )

            if status == "INSERTED":
                inserted += 1
            else:
                updated += 1

        connection.commit()

        benchmark_parameters = sorted(
            expected_benchmark_ids
        )
        benchmark_markers = ", ".join(
            "?"
            for _ in benchmark_parameters
        )

        cursor.execute(
            f"""
            SELECT
                benchmark_id,
                orchestration_pattern,
                source_lab,
                outcome_valid,
                model_calls,
                input_tokens,
                output_tokens,
                latency_ms,
                estimated_model_cost_usd,
                external_action_executed
            FROM {BENCHMARK_TABLE}
            WHERE benchmark_id IN (
                {benchmark_markers}
            )
            ORDER BY benchmark_id
            """,
            benchmark_parameters,
        )
        stored_benchmarks = (
            cursor.fetchall()
        )

        event_parameters = sorted(
            expected_event_ids
        )
        event_markers = ", ".join(
            "?"
            for _ in event_parameters
        )

        cursor.execute(
            f"""
            SELECT
                event_id,
                external_action_executed
            FROM {LEDGER_TABLE}
            WHERE event_id IN (
                {event_markers}
            )
            """,
            event_parameters,
        )
        stored_events = cursor.fetchall()

        cursor.execute(
            f"""
            SELECT
                route_name,
                SUM(model_calls)
                    AS model_calls,
                SUM(input_tokens)
                    AS input_tokens,
                SUM(output_tokens)
                    AS output_tokens,
                ROUND(
                    SUM(
                        estimated_model_cost_usd
                    ),
                    8
                ) AS estimated_cost_usd,
                ROUND(
                    AVG(latency_ms),
                    2
                ) AS average_latency_ms
            FROM {LEDGER_TABLE}
            WHERE event_id IN (
                {event_markers}
            )
            GROUP BY route_name
            ORDER BY route_name
            """,
            event_parameters,
        )
        finops_summary = cursor.fetchall()

actual_benchmark_ids = {
    str(row[0])
    for row in stored_benchmarks
}
actual_event_ids = {
    str(row[0])
    for row in stored_events
}

benchmarks_valid = (
    actual_benchmark_ids
    == expected_benchmark_ids
    and all(
        row[3] is True
        and row[9] is False
        for row in stored_benchmarks
    )
)

ledger_valid = (
    actual_event_ids
    == expected_event_ids
    and all(
        row[1] is False
        for row in stored_events
    )
)

print("\nSTORED BENCHMARKS:")

for row in stored_benchmarks:
    print(
        {
            "benchmark_id": row[0],
            "pattern": row[1],
            "source_lab": row[2],
            "outcome_valid": row[3],
            "model_calls": row[4],
            "input_tokens": row[5],
            "output_tokens": row[6],
            "latency_ms": row[7],
            "estimated_cost_usd": row[8],
            "external_action": row[9],
        }
    )

print("\nFINOPS SUMMARY:")

for row in finops_summary:
    print(
        {
            "route_name": row[0],
            "model_calls": row[1],
            "input_tokens": row[2],
            "output_tokens": row[3],
            "estimated_cost_usd": row[4],
            "average_latency_ms": row[5],
        }
    )

print("\nROWS INSERTED:", inserted)
print("ROWS UPDATED:", updated)
print("ROWS DELETED: 0")
print(
    "BENCHMARKS VERIFIED:",
    benchmarks_valid,
)
print(
    "USAGE LEDGER VERIFIED:",
    ledger_valid,
)
print(
    "DATABRICKS METADATA WRITES:",
    inserted + updated,
)
print("UNDERWRITING ACTIONS EXECUTED: 0")
print("SECRETS PRINTED: False")

if not (
    benchmarks_valid
    and ledger_valid
):
    raise RuntimeError(
        "Databricks metrics validation failed."
    )

print("LAB 10 METRICS LOAD: COMPLETE")
