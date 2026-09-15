"""Verify completed CASE-001 state through governed MCP tools."""

import json
from typing import Any

import anyio
from mcp import Client
from mcp.types import TextContent

from v2.mcp_server.underwriting_server import mcp
from v2.services.telemetry import record_event


EXPECTED_TOOLS = {
    "submission",
    "weather_risk",
    "compliance_rules",
    "loss_history",
    "base_score",
}


def extract_payload(
    result: Any,
) -> dict[str, Any]:
    """Extract MCP structured or model-facing JSON."""

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

        if not isinstance(candidate, dict):
            continue

        payload = candidate.get(
            "result",
            candidate,
        )

        if isinstance(payload, dict):
            return payload

    raise ValueError(
        "MCP result contained no JSON object."
    )


async def main() -> None:
    async with Client(
        mcp,
        raise_exceptions=True,
    ) as client:
        print("MCP CONNECTION: SUCCESS")
        print(
            "SERVER NAME:",
            getattr(
                client.server_info,
                "name",
                None,
            ),
        )
        print(
            "PROTOCOL VERSION:",
            getattr(
                client.server_info,
                "protocol_version",
                None,
            ),
        )

        listed = await client.list_tools()
        discovered_tools = {
            tool.name
            for tool in listed.tools
        }

        print(
            "DISCOVERED TOOLS:",
            sorted(discovered_tools),
        )

        submission_result = await client.call_tool(
            "submission",
            {"case_id": "CASE-001"},
        )
        compliance_result = await client.call_tool(
            "compliance_rules",
            {"case_id": "CASE-001"},
        )
        loss_result = await client.call_tool(
            "loss_history",
            {"case_id": "CASE-001"},
        )

        results_have_no_errors = all(
            getattr(result, "is_error", True)
            is False
            for result in (
                submission_result,
                compliance_result,
                loss_result,
            )
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

        state_summary = {
            "submission": {
                "case_id": (
                    submission_payload.get(
                        "case_id"
                    )
                ),
                "evidence_status": (
                    submission_payload.get(
                        "evidence_status"
                    )
                ),
                "outstanding_evidence": (
                    submission_payload.get(
                        "outstanding_evidence"
                    )
                ),
            },
            "compliance": {
                "policy_id": (
                    compliance_payload.get(
                        "policy_id"
                    )
                ),
                "evidence_status": (
                    compliance_payload.get(
                        "evidence_status"
                    )
                ),
                "next_step": (
                    compliance_payload.get(
                        "next_step"
                    )
                ),
                "human_review_required": (
                    compliance_payload.get(
                        "human_review_required"
                    )
                ),
                "human_approval_required": (
                    compliance_payload.get(
                        "human_approval_required"
                    )
                ),
            },
            "loss_history": {
                "evidence_status": (
                    loss_payload.get(
                        "evidence_status"
                    )
                ),
                "received": (
                    loss_payload.get("received")
                ),
                "verified": (
                    loss_payload.get("verified")
                ),
                "record_count": len(
                    loss_payload.get(
                        "records",
                        [],
                    )
                ),
            },
        }

        tools_valid = (
            discovered_tools == EXPECTED_TOOLS
        )

        state_valid = all(
            (
                submission_payload.get(
                    "case_id"
                )
                == "CASE-001",
                submission_payload.get(
                    "evidence_status"
                )
                == "COMPLETE",
                submission_payload.get(
                    "outstanding_evidence"
                )
                == [],
                compliance_payload.get(
                    "evidence_status"
                )
                == "COMPLETE",
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
                len(
                    loss_payload.get(
                        "records",
                        [],
                    )
                )
                == 2,
            )
        )

        verified = all(
            (
                tools_valid,
                results_have_no_errors,
                state_valid,
            )
        )

        record_event(
            {
                "lab": "V2_CAPSTONE",
                "event_type": (
                    "MCP_SHARED_STATE_VERIFICATION"
                ),
                "case_id": "CASE-001",
                "mcp_tool_calls": 3,
                "tools_used": [
                    "submission",
                    "compliance_rules",
                    "loss_history",
                ],
                "evidence_status": (
                    submission_payload.get(
                        "evidence_status"
                    )
                ),
                "policy_next_step": (
                    compliance_payload.get(
                        "next_step"
                    )
                ),
                "loss_history_status": (
                    loss_payload.get(
                        "evidence_status"
                    )
                ),
                "model_calls": 0,
                "external_action_executed": False,
                "coverage_decision_executed": False,
                "verified": verified,
            }
        )

        print("\nMCP STATE SUMMARY:")
        print(
            json.dumps(
                state_summary,
                indent=2,
            )
        )

        print(
            "\nEXACT APPROVED TOOL SET:",
            tools_valid,
        )
        print(
            "MCP RESULTS HAVE NO ERRORS:",
            results_have_no_errors,
        )
        print(
            "MCP COMPLETED STATE VALID:",
            state_valid,
        )
        print(
            "MCP SHARED STATE VERIFIED:",
            verified,
        )
        print("MCP TOOL CALLS: 3")
        print("CLAUDE API CALLS: 0")
        print("EXTERNAL ACTIONS EXECUTED: 0")
        print("SECRETS PRINTED: False")

        if not verified:
            raise RuntimeError(
                "MCP shared-state validation failed."
            )

        print(
            "CAPSTONE MCP STATE CHECK: COMPLETE"
        )


if __name__ == "__main__":
    anyio.run(main)
