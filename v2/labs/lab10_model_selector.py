import json
import os
from pathlib import Path

from dotenv import load_dotenv

from v2.control_plane.model_selector import (
    load_model_catalog,
    select_model,
)
from v2.services.telemetry import record_event


PROJECT_ROOT = Path(__file__).resolve().parents[2]

load_dotenv(PROJECT_ROOT / ".env")

configured_model = os.getenv(
    "CLAUDE_MODEL",
    "",
).strip()

catalog_entries = load_model_catalog()

print(
    "MODEL CATALOG ROWS LOADED:",
    len(catalog_entries),
)

no_llm = select_model(
    "NO_LLM",
    catalog_entries,
)

conversational = select_model(
    "CONVERSATIONAL",
    catalog_entries,
    expected_input_tokens=500,
    expected_output_tokens=100,
)

premium = select_model(
    "PREMIUM_REASONING",
    catalog_entries,
    expected_input_tokens=1000,
    expected_output_tokens=300,
)

consequential = select_model(
    "CONVERSATIONAL",
    catalog_entries,
    expected_input_tokens=500,
    expected_output_tokens=100,
    consequential_action_requested=True,
)

invalid = select_model(
    "UNKNOWN_ROUTE",
    catalog_entries,
)

results = {
    "no_llm": no_llm,
    "conversational": conversational,
    "premium": premium,
    "consequential_request": consequential,
    "invalid_route": invalid,
}

print("\nSELECTOR RESULTS:")
print(json.dumps(results, indent=2))

no_llm_valid = all(
    (
        no_llm["selection_status"]
        == "SELECTED",
        no_llm["model_id"] == "NO_LLM",
        no_llm[
            "estimated_model_cost_usd"
        ]
        == 0.0,
    )
)

conversational_valid = all(
    (
        conversational["selection_status"]
        == "SELECTED",
        conversational["model_id"]
        == configured_model,
        conversational[
            "governance_status"
        ]
        == "QUALIFIED_FOR_TRAINING",
        conversational[
            "estimated_model_cost_usd"
        ]
        == 0.001,
    )
)

premium_fail_closed = all(
    (
        premium["selection_status"]
        == "NO_QUALIFIED_MODEL",
        premium["model_id"] is None,
        premium["reason_code"]
        == "FAIL_CLOSED_CATALOG_FILTER",
    )
)

consequential_blocked = all(
    (
        consequential["selection_status"]
        == "BLOCKED",
        consequential["model_id"] is None,
        consequential["reason_code"]
        == (
            "CONSEQUENTIAL_ACTION_REQUIRES_"
            "APPLICATION_AUTHORIZATION"
        ),
    )
)

invalid_blocked = all(
    (
        invalid["selection_status"]
        == "INVALID_ROUTE",
        invalid["model_id"] is None,
    )
)

all_results_safe = all(
    (
        result[
            "model_call_executed"
        ]
        is False,
        result[
            "external_action_executed"
        ]
        is False,
        result[
            "coverage_decision_executed"
        ]
        is False,
        result[
            "allowed_for_"
            "consequential_decisions"
        ]
        is False,
    )
    for result in results.values()
)

selector_source = (
    PROJECT_ROOT
    / "v2"
    / "control_plane"
    / "model_selector.py"
).read_text(encoding="utf-8")

model_id_hard_coded = (
    configured_model in selector_source
)

selector_verified = all(
    (
        len(catalog_entries) >= 3,
        no_llm_valid,
        conversational_valid,
        premium_fail_closed,
        consequential_blocked,
        invalid_blocked,
        all_results_safe,
        not model_id_hard_coded,
    )
)

report = {
    "catalog_rows_loaded": len(
        catalog_entries
    ),
    "no_llm_valid": no_llm_valid,
    "conversational_valid": (
        conversational_valid
    ),
    "premium_fail_closed": (
        premium_fail_closed
    ),
    "consequential_request_blocked": (
        consequential_blocked
    ),
    "invalid_route_blocked": (
        invalid_blocked
    ),
    "all_results_safe": all_results_safe,
    "model_id_hard_coded": (
        model_id_hard_coded
    ),
    "claude_api_calls": 0,
    "databricks_reads": 1,
    "databricks_writes": 0,
    "underwriting_actions_executed": 0,
    "selector_verified": selector_verified,
}

results_file = (
    PROJECT_ROOT
    / "v2"
    / "data"
    / "lab10_selector_results.json"
)
results_file.write_text(
    json.dumps(
        {
            "report": report,
            "results": results,
        },
        indent=2,
    ),
    encoding="utf-8",
)

record_event(
    {
        "lab": "V2_LAB_10",
        "route": "MODEL_CATALOG_SELECTOR",
        "model_id": "NO_MODEL_CALLED",
        "input_tokens": 0,
        "output_tokens": 0,
        "model_calls": 0,
        "tool_calls": 0,
        "databricks_reads": 1,
        "stop_reason": (
            "DETERMINISTIC_COMPLETE"
        ),
        "premium_fail_closed": (
            premium_fail_closed
        ),
        "consequential_request_blocked": (
            consequential_blocked
        ),
        "external_action_executed": False,
        "coverage_decision_executed": False,
        "lab_verified": selector_verified,
    }
)

print("\nLAB 10 REPORT:")
print(json.dumps(report, indent=2))

print(
    "\nRESULT FILE:",
    results_file.name,
)
print("SECRETS PRINTED: False")

if not selector_verified:
    raise RuntimeError(
        "Lab 10 selector validation failed. "
        "Paste the complete output."
    )

print("V2 LAB 10 STATUS: COMPLETE")
