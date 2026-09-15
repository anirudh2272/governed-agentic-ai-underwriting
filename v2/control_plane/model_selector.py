import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from databricks import sql
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_CATALOG_TABLE = (
    "workspace.ai_control_plane.model_catalog"
)

VALID_ROUTES = frozenset(
    {
        "NO_LLM",
        "CONVERSATIONAL",
        "PREMIUM_REASONING",
    }
)

QUALIFIED_STATUS = "QUALIFIED_FOR_TRAINING"


def load_model_catalog() -> list[dict[str, Any]]:
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

    if not hostname or not http_path:
        raise RuntimeError(
            "Databricks configuration is missing."
        )

    if auth_type != "databricks-oauth":
        raise RuntimeError(
            "Expected databricks-oauth."
        )

    query = f"""
    SELECT
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
        notes
    FROM {MODEL_CATALOG_TABLE}
    ORDER BY catalog_entry_id
    """

    with sql.connect(
        server_hostname=hostname,
        http_path=http_path,
        auth_type=auth_type,
        use_cloud_fetch=False,
    ) as connection:
        with connection.cursor() as cursor:
            cursor.execute(query)

            column_names = [
                column[0]
                for column in cursor.description
            ]
            rows = cursor.fetchall()

    return [
        dict(zip(column_names, row))
        for row in rows
    ]


def _base_result(
    requested_route: str,
    status: str,
    reason_code: str,
) -> dict[str, Any]:
    return {
        "requested_route": requested_route,
        "selection_status": status,
        "reason_code": reason_code,
        "catalog_entry_id": None,
        "model_id": None,
        "provider": None,
        "capability_tier": None,
        "governance_status": None,
        "estimated_model_cost_usd": None,
        "qualification_sample_size": None,
        "allowed_for_consequential_decisions": (
            False
        ),
        "model_call_executed": False,
        "external_action_executed": False,
        "coverage_decision_executed": False,
    }


def _estimated_cost(
    entry: Mapping[str, Any],
    expected_input_tokens: int,
    expected_output_tokens: int,
) -> float:
    input_rate = float(
        entry[
            "input_cost_usd_per_million"
        ]
    )
    output_rate = float(
        entry[
            "output_cost_usd_per_million"
        ]
    )

    cost = (
        expected_input_tokens
        / 1_000_000
        * input_rate
        + expected_output_tokens
        / 1_000_000
        * output_rate
    )

    return round(cost, 8)


def select_model(
    requested_route: str,
    catalog_entries: Iterable[
        Mapping[str, Any]
    ],
    *,
    expected_input_tokens: int = 0,
    expected_output_tokens: int = 0,
    consequential_action_requested: bool = False,
) -> dict[str, Any]:
    normalized_route = (
        str(requested_route).strip().upper()
    )

    if normalized_route not in VALID_ROUTES:
        return _base_result(
            normalized_route,
            "INVALID_ROUTE",
            "ROUTE_NOT_SUPPORTED",
        )

    valid_token_estimates = all(
        (
            isinstance(
                expected_input_tokens,
                int,
            ),
            not isinstance(
                expected_input_tokens,
                bool,
            ),
            expected_input_tokens >= 0,
            isinstance(
                expected_output_tokens,
                int,
            ),
            not isinstance(
                expected_output_tokens,
                bool,
            ),
            expected_output_tokens >= 0,
        )
    )

    if not valid_token_estimates:
        return _base_result(
            normalized_route,
            "INVALID_REQUEST",
            "INVALID_TOKEN_ESTIMATE",
        )

    if consequential_action_requested:
        return _base_result(
            normalized_route,
            "BLOCKED",
            (
                "CONSEQUENTIAL_ACTION_REQUIRES_"
                "APPLICATION_AUTHORIZATION"
            ),
        )

    qualified_candidates = [
        dict(entry)
        for entry in catalog_entries
        if (
            str(
                entry.get("route_name", "")
            ).upper()
            == normalized_route
            and entry.get("active") is True
            and str(
                entry.get(
                    "governance_status",
                    "",
                )
            ).upper()
            == QUALIFIED_STATUS
            and entry.get(
                "allowed_for_"
                "consequential_decisions"
            )
            is False
        )
    ]

    if not qualified_candidates:
        return _base_result(
            normalized_route,
            "NO_QUALIFIED_MODEL",
            "FAIL_CLOSED_CATALOG_FILTER",
        )

    qualified_candidates.sort(
        key=lambda entry: (
            _estimated_cost(
                entry,
                expected_input_tokens,
                expected_output_tokens,
            ),
            -int(
                entry.get(
                    "qualification_sample_size",
                    0,
                )
            ),
            str(
                entry.get(
                    "catalog_entry_id",
                    "",
                )
            ),
        )
    )

    selected = qualified_candidates[0]

    result = _base_result(
        normalized_route,
        "SELECTED",
        "QUALIFIED_LOWEST_ESTIMATED_COST",
    )

    result.update(
        {
            "catalog_entry_id": selected[
                "catalog_entry_id"
            ],
            "model_id": selected["model_id"],
            "provider": selected["provider"],
            "capability_tier": selected[
                "capability_tier"
            ],
            "governance_status": selected[
                "governance_status"
            ],
            "estimated_model_cost_usd": (
                _estimated_cost(
                    selected,
                    expected_input_tokens,
                    expected_output_tokens,
                )
            ),
            "qualification_sample_size": (
                int(
                    selected[
                        "qualification_sample_size"
                    ]
                )
            ),
            "allowed_for_consequential_decisions": (
                bool(
                    selected[
                        "allowed_for_"
                        "consequential_decisions"
                    ]
                )
            ),
        }
    )

    return result
