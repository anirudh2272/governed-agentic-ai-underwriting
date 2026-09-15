import os
from datetime import date
from pathlib import Path

from databricks import sql
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DDL_FILE = (
    PROJECT_ROOT
    / "v2"
    / "sql"
    / "10_control_plane_catalog.sql"
)

CATALOG_TABLE = (
    "workspace.ai_control_plane.model_catalog"
)

PRICE_SOURCE = (
    "https://docs.anthropic.com/"
    "en/docs/about-claude/pricing"
)


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

if configured_model not in {
    "claude-haiku-4-5",
    "claude-haiku-4-5-20251001",
}:
    raise SystemExit(
        "This setup contains verified pricing "
        "only for Claude Haiku 4.5."
    )

entries = [
    {
        "catalog_entry_id": (
            "no_llm_deterministic"
        ),
        "route_name": "NO_LLM",
        "model_id": "NO_LLM",
        "provider": "APPLICATION",
        "capability_tier": "DETERMINISTIC",
        "governance_status": (
            "QUALIFIED_FOR_TRAINING"
        ),
        "active": True,
        "allowed_for_consequential_decisions": (
            False
        ),
        "input_cost": 0.0,
        "output_cost": 0.0,
        "sample_size": 1,
        "evidence_source": "V2_LAB_04",
        "price_source": "NOT_APPLICABLE",
        "price_date": date(2026, 9, 15),
        "notes": (
            "Exact policy outcome without a "
            "model call"
        ),
    },
    {
        "catalog_entry_id": (
            "haiku_45_conversational"
        ),
        "route_name": "CONVERSATIONAL",
        "model_id": configured_model,
        "provider": "ANTHROPIC",
        "capability_tier": "CONVERSATIONAL",
        "governance_status": (
            "QUALIFIED_FOR_TRAINING"
        ),
        "active": True,
        "allowed_for_consequential_decisions": (
            False
        ),
        "input_cost": 1.0,
        "output_cost": 5.0,
        "sample_size": 1,
        "evidence_source": (
            "V2_LAB_05_AND_09"
        ),
        "price_source": PRICE_SOURCE,
        "price_date": date(2026, 9, 15),
        "notes": (
            "Training-only qualification based "
            "on one synthetic case"
        ),
    },
    {
        "catalog_entry_id": (
            "haiku_45_premium_candidate"
        ),
        "route_name": "PREMIUM_REASONING",
        "model_id": configured_model,
        "provider": "ANTHROPIC",
        "capability_tier": (
            "PREMIUM_REASONING"
        ),
        "governance_status": (
            "EVALUATION_REQUIRED"
        ),
        "active": False,
        "allowed_for_consequential_decisions": (
            False
        ),
        "input_cost": 1.0,
        "output_cost": 5.0,
        "sample_size": 0,
        "evidence_source": "V2_LAB_05",
        "price_source": PRICE_SOURCE,
        "price_date": date(2026, 9, 15),
        "notes": (
            "Physical model was exercised, but "
            "premium capability was not qualified"
        ),
    },
]

ddl_statements = [
    statement.strip()
    for statement in DDL_FILE.read_text(
        encoding="utf-8"
    ).split(";")
    if statement.strip()
]

select_existing_sql = f"""
SELECT COUNT(*)
FROM {CATALOG_TABLE}
WHERE catalog_entry_id = ?
"""

insert_sql = f"""
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
VALUES (
    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
    current_timestamp()
)
"""

update_sql = f"""
UPDATE {CATALOG_TABLE}
SET
    route_name = ?,
    model_id = ?,
    provider = ?,
    capability_tier = ?,
    governance_status = ?,
    active = ?,
    allowed_for_consequential_decisions = ?,
    input_cost_usd_per_million = ?,
    output_cost_usd_per_million = ?,
    qualification_sample_size = ?,
    evidence_source = ?,
    price_source = ?,
    price_verified_date = ?,
    notes = ?,
    updated_at = current_timestamp()
WHERE catalog_entry_id = ?
"""


def entry_values(entry):
    return [
        entry["route_name"],
        entry["model_id"],
        entry["provider"],
        entry["capability_tier"],
        entry["governance_status"],
        entry["active"],
        entry[
            "allowed_for_consequential_decisions"
        ],
        entry["input_cost"],
        entry["output_cost"],
        entry["sample_size"],
        entry["evidence_source"],
        entry["price_source"],
        entry["price_date"],
        entry["notes"],
    ]


print("OPENING DATABRICKS CONNECTION...")

inserted = 0
updated = 0

with sql.connect(
    server_hostname=hostname,
    http_path=http_path,
    auth_type=auth_type,
    use_cloud_fetch=False,
) as connection:
    with connection.cursor() as cursor:
        for statement in ddl_statements:
            cursor.execute(statement)

        print(
            "SCHEMA AND TABLE DEFINITIONS: "
            "EXECUTED"
        )

        for entry in entries:
            cursor.execute(
                select_existing_sql,
                [entry["catalog_entry_id"]],
            )
            exists = cursor.fetchone()[0] > 0

            values = entry_values(entry)

            if exists:
                cursor.execute(
                    update_sql,
                    values
                    + [
                        entry[
                            "catalog_entry_id"
                        ]
                    ],
                )
                updated += 1
            else:
                cursor.execute(
                    insert_sql,
                    [
                        entry[
                            "catalog_entry_id"
                        ],
                        *values,
                    ],
                )
                inserted += 1

        connection.commit()

        cursor.execute(
            """
            SHOW TABLES IN
            workspace.ai_control_plane
            """
        )
        table_rows = cursor.fetchall()
        table_names = {
            str(row[1])
            for row in table_rows
        }

        cursor.execute(
            f"""
            SELECT
                catalog_entry_id,
                route_name,
                model_id,
                governance_status,
                active,
                allowed_for_consequential_decisions,
                input_cost_usd_per_million,
                output_cost_usd_per_million,
                qualification_sample_size
            FROM {CATALOG_TABLE}
            ORDER BY catalog_entry_id
            """
        )
        catalog_rows = cursor.fetchall()

required_tables = {
    "model_catalog",
    "model_benchmarks",
    "model_usage_ledger",
}
expected_entries = {
    entry["catalog_entry_id"]
    for entry in entries
}
actual_entries = {
    str(row[0])
    for row in catalog_rows
}

all_consequential_actions_blocked = all(
    row[5] is False
    for row in catalog_rows
    if row[0] in expected_entries
)

print("\nTABLES:")
for table_name in sorted(table_names):
    print("-", table_name)

print("\nMODEL CATALOG:")
for row in catalog_rows:
    print(
        {
            "catalog_entry_id": row[0],
            "route_name": row[1],
            "model_id": row[2],
            "governance_status": row[3],
            "active": row[4],
            "consequential_decisions": row[5],
            "input_cost_per_million": row[6],
            "output_cost_per_million": row[7],
            "qualification_sample_size": row[8],
        }
    )

tables_verified = (
    required_tables.issubset(table_names)
)
entries_verified = (
    expected_entries.issubset(actual_entries)
)

print("\nROWS INSERTED:", inserted)
print("ROWS UPDATED:", updated)
print("ROWS DELETED: 0")
print(
    "TABLES VERIFIED:",
    tables_verified,
)
print(
    "CATALOG ENTRIES VERIFIED:",
    entries_verified,
)
print(
    "CONSEQUENTIAL DECISIONS DISALLOWED:",
    all_consequential_actions_blocked,
)
print("SECRETS PRINTED: False")

if not all(
    (
        tables_verified,
        entries_verified,
        all_consequential_actions_blocked,
    )
):
    raise RuntimeError(
        "Model catalog setup validation failed."
    )

print("LAB 10 CATALOG SETUP: COMPLETE")
