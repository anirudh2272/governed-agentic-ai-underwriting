"""Zero-model preflight for Capstone Phase B."""

import json
from pathlib import Path

import anyio
from mcp import Client

from tools.underwriting_tools import (
    compliance_rules,
    loss_history,
    submission,
)
from v2.control_plane.model_selector import (
    load_model_catalog,
    select_model,
)
from v2.control_plane.policies import (
    get_route_policy,
    is_data_classification_allowed,
)
from v2.control_plane.router import route_request
from v2.control_plane.tool_gateway import (
    DEFAULT_ALLOWED_TOOLS,
)
from v2.control_plane.tool_schemas import (
    build_claude_tool_definitions,
)
from v2.mcp_server.underwriting_server import mcp
from v2.services.telemetry import record_event


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULT_FILE = (
    PROJECT_ROOT
    / "v2"
    / "data"
    / "capstone_phase_b_preflight.json"
)

EXPECTED_CATALOG_ENTRY = (
    "haiku_45_premium_capstone"
)

FORBIDDEN_MODEL_TOOLS = {
    "authorize_underwriting_step",
    "bind_coverage",
    "approve",
    "decline",
    "set_premium",
}


async def main() -> None:
    case = submission("CASE-001")
    compliance = compliance_rules("CASE-001")
    losses = loss_history("CASE-001")

    state_valid = all(
        (
            case.get("evidence_status")
            == "COMPLETE",
            case.get("outstanding_evidence")
            == [],
            compliance.get("next_step")
            == "HUMAN_REVIEW",
            compliance.get(
                "human_review_required"
            )
            is True,
            compliance.get(
                "human_approval_required"
            )
            is True,
            losses.get("evidence_status")
            == "VERIFIED",
            len(losses.get("records", []))
            == 2,
        )
    )

    decision = route_request(
        task_type="UNDERWRITING_RECOMMENDATION",
        deterministic_answer=None,
        complexity="HIGH",
        business_risk="HIGH",
    )

    route_valid = all(
        (
            decision.get("route")
            == "PREMIUM_REASONING",
            decision.get("model_required")
            is True,
        )
    )

    policy = get_route_policy(
        decision["route"]
    )

    data_allowed = (
        is_data_classification_allowed(
            decision["route"],
            "SYNTHETIC_TRAINING",
        )
    )

    print("OPENING DATABRICKS MODEL CATALOG...")

    catalog = load_model_catalog()

    selection = select_model(
        decision["route"],
        catalog,
        expected_input_tokens=3000,
        expected_output_tokens=500,
        consequential_action_requested=False,
    )

    selection_valid = all(
        (
            selection.get(
                "selection_status"
            )
            == "SELECTED",
            selection.get(
                "catalog_entry_id"
            )
            == EXPECTED_CATALOG_ENTRY,
            selection.get("model_id")
            is not None,
            selection.get(
                "allowed_for_"
                "consequential_decisions"
            )
            is False,
        )
    )

    estimated_cost = float(
        selection.get(
            "estimated_model_cost_usd",
            999,
        )
    )
    cost_limit = float(
        policy[
            "max_estimated_model_cost_usd"
        ]
    )

    cost_policy_valid = (
        estimated_cost <= cost_limit
    )

    allowed_tools = frozenset(
        policy["allowed_tools"]
    )

    tool_definitions = (
        build_claude_tool_definitions(
            allowed_tools
        )
    )
    definition_names = {
        definition["name"]
        for definition in tool_definitions
    }

    async with Client(
        mcp,
        raise_exceptions=True,
    ) as client:
        listed = await client.list_tools()
        server_tools = {
            tool.name
            for tool in listed.tools
        }

    tool_policy_valid = all(
        (
            allowed_tools
            == DEFAULT_ALLOWED_TOOLS,
            definition_names
            == DEFAULT_ALLOWED_TOOLS,
            server_tools
            == DEFAULT_ALLOWED_TOOLS,
            not (
                server_tools
                & FORBIDDEN_MODEL_TOOLS
            ),
            policy["max_agent_steps"] == 6,
        )
    )

    preflight_verified = all(
        (
            state_valid,
            route_valid,
            data_allowed,
            selection_valid,
            cost_policy_valid,
            tool_policy_valid,
        )
    )

    report = {
        "case_state": {
            "case_id": case.get("case_id"),
            "evidence_status": case.get(
                "evidence_status"
            ),
            "policy_next_step": (
                compliance.get("next_step")
            ),
            "loss_history_status": (
                losses.get(
                    "evidence_status"
                )
            ),
            "loss_record_count": len(
                losses.get("records", [])
            ),
        },
        "route_decision": decision,
        "data_classification": (
            "SYNTHETIC_TRAINING"
        ),
        "data_classification_allowed": (
            data_allowed
        ),
        "model_selection": selection,
        "cost_policy": {
            "estimated_model_cost_usd": (
                estimated_cost
            ),
            "maximum_model_cost_usd": (
                cost_limit
            ),
            "within_limit": (
                cost_policy_valid
            ),
        },
        "tool_policy": {
            "allowed_tools": sorted(
                allowed_tools
            ),
            "claude_tool_definitions": sorted(
                definition_names
            ),
            "mcp_server_tools": sorted(
                server_tools
            ),
            "forbidden_tools_exposed": sorted(
                server_tools
                & FORBIDDEN_MODEL_TOOLS
            ),
            "max_agent_steps": policy[
                "max_agent_steps"
            ],
        },
        "state_valid": state_valid,
        "route_valid": route_valid,
        "selection_valid": selection_valid,
        "tool_policy_valid": (
            tool_policy_valid
        ),
        "preflight_verified": (
            preflight_verified
        ),
        "claude_api_calls": 0,
        "mcp_tool_calls": 0,
        "external_actions_executed": 0,
    }

    RESULT_FILE.write_text(
        json.dumps(
            report,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    record_event(
        {
            "lab": "V2_CAPSTONE",
            "event_type": (
                "PHASE_B_PREFLIGHT"
            ),
            "case_id": "CASE-001",
            "route": decision.get("route"),
            "model_id": selection.get(
                "model_id"
            ),
            "catalog_entry_id": (
                selection.get(
                    "catalog_entry_id"
                )
            ),
            "data_classification": (
                "SYNTHETIC_TRAINING"
            ),
            "estimated_model_cost_usd": (
                estimated_cost
            ),
            "maximum_model_cost_usd": (
                cost_limit
            ),
            "allowed_tools": sorted(
                allowed_tools
            ),
            "max_agent_steps": policy[
                "max_agent_steps"
            ],
            "model_calls": 0,
            "external_action_executed": False,
            "coverage_decision_executed": False,
            "verified": preflight_verified,
        }
    )

    print("\nCASE STATE:")
    print(
        json.dumps(
            report["case_state"],
            indent=2,
        )
    )

    print("\nROUTE DECISION:")
    print(json.dumps(decision, indent=2))

    print("\nMODEL SELECTION:")
    print(json.dumps(selection, indent=2))

    print("\nCOST POLICY:")
    print(
        json.dumps(
            report["cost_policy"],
            indent=2,
        )
    )

    print("\nTOOL POLICY:")
    print(
        json.dumps(
            report["tool_policy"],
            indent=2,
        )
    )

    print(
        "\nSTATE VALID:",
        state_valid,
    )
    print(
        "PREMIUM ROUTE VALID:",
        route_valid,
    )
    print(
        "DATA CLASSIFICATION ALLOWED:",
        data_allowed,
    )
    print(
        "DYNAMIC MODEL SELECTION VALID:",
        selection_valid,
    )
    print(
        "COST POLICY VALID:",
        cost_policy_valid,
    )
    print(
        "MCP TOOL POLICY VALID:",
        tool_policy_valid,
    )
    print(
        "PHASE B PREFLIGHT VERIFIED:",
        preflight_verified,
    )
    print("CLAUDE API CALLS: 0")
    print("MCP TOOL CALLS: 0")
    print("UNDERWRITING ACTIONS EXECUTED: 0")
    print("SECRETS PRINTED: False")

    if not preflight_verified:
        raise RuntimeError(
            "Capstone Phase B preflight failed."
        )

    print(
        "CAPSTONE PHASE B PREFLIGHT: COMPLETE"
    )


if __name__ == "__main__":
    anyio.run(main)
