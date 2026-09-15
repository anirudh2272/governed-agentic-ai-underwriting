import json
from typing import Any

import anyio
from mcp import Client
from mcp.types import TextContent

from v2.mcp_server.underwriting_server import mcp


EXPECTED_ARGUMENTS = {
    "submission": {"case_id"},
    "weather_risk": {"location"},
    "compliance_rules": {"case_id"},
    "loss_history": {"case_id"},
    "base_score": {"case_id"},
}

EXPECTED_TOOLS = set(EXPECTED_ARGUMENTS)

FORBIDDEN_TOOLS = {
    "authorize_underwriting_step",
    "bind_coverage",
    "approve",
    "decline",
    "set_premium",
}


def extract_payload(result: Any) -> dict[str, Any]:
    """Read structured MCP output, with text-content fallback."""

    structured = getattr(
        result,
        "structured_content",
        None,
    )

    if isinstance(structured, dict):
        payload = structured.get("result", structured)

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
            payload = candidate.get("result", candidate)

            if isinstance(payload, dict):
                return payload

    raise RuntimeError(
        "No usable dictionary payload returned by MCP."
    )


async def main() -> None:
    async with Client(
        mcp,
        raise_exceptions=True,
    ) as client:
        print("MCP CONNECTION: SUCCESS")
        print(
            "SERVER NAME:",
            getattr(client.server_info, "name", None),
        )
        print(
            "PROTOCOL VERSION:",
            client.protocol_version,
        )

        listed = await client.list_tools()
        tool_names = {
            tool.name
            for tool in listed.tools
        }

        print(
            "\nDISCOVERED TOOLS:",
            sorted(tool_names),
        )

        schemas_valid = True

        print("\nDISCOVERED INPUT SCHEMAS:")

        for tool in sorted(
            listed.tools,
            key=lambda item: item.name,
        ):
            properties = set(
                tool.input_schema.get(
                    "properties",
                    {},
                )
            )
            required = set(
                tool.input_schema.get(
                    "required",
                    [],
                )
            )
            expected = EXPECTED_ARGUMENTS.get(
                tool.name,
                set(),
            )

            schema_valid = (
                properties == expected
                and required == expected
            )
            schemas_valid = (
                schemas_valid
                and schema_valid
            )

            print(
                f"- {tool.name}: "
                f"arguments={sorted(properties)}, "
                f"valid={schema_valid}"
            )

        exposed_forbidden = (
            tool_names.intersection(
                FORBIDDEN_TOOLS
            )
        )

        submission_result = (
            await client.call_tool(
                "submission",
                {"case_id": "CASE-001"},
            )
        )
        compliance_result = (
            await client.call_tool(
                "compliance_rules",
                {"case_id": "CASE-001"},
            )
        )

        submission_payload = extract_payload(
            submission_result
        )
        compliance_payload = extract_payload(
            compliance_result
        )

        submission_summary = {
            "case_id": submission_payload.get(
                "case_id"
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
        }

        compliance_summary = {
            "case_id": compliance_payload.get(
                "case_id"
            ),
            "policy_id": compliance_payload.get(
                "policy_id"
            ),
            "next_step": compliance_payload.get(
                "next_step"
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
        }

        print(
            "\nSUBMISSION MCP ERROR:",
            bool(submission_result.is_error),
        )
        print(
            "SUBMISSION RESULT:",
            json.dumps(
                submission_summary,
                indent=2,
            ),
        )

        print(
            "\nCOMPLIANCE MCP ERROR:",
            bool(compliance_result.is_error),
        )
        print(
            "COMPLIANCE RESULT:",
            json.dumps(
                compliance_summary,
                indent=2,
            ),
        )

        verified = all(
            (
                tool_names == EXPECTED_TOOLS,
                schemas_valid,
                not exposed_forbidden,
                not submission_result.is_error,
                not compliance_result.is_error,
                (
                    submission_summary[
                        "evidence_status"
                    ]
                    == "INCOMPLETE"
                ),
                (
                    compliance_summary["next_step"]
                    == "REQUEST_EVIDENCE"
                ),
                (
                    compliance_summary[
                        "human_review_required"
                    ]
                    is True
                ),
                (
                    compliance_summary[
                        "human_approval_required"
                    ]
                    is True
                ),
            )
        )

        print(
            "\nEXACT APPROVED TOOL SET:",
            tool_names == EXPECTED_TOOLS,
        )
        print(
            "INPUT SCHEMAS VALID:",
            schemas_valid,
        )
        print(
            "FORBIDDEN TOOLS EXPOSED:",
            sorted(exposed_forbidden),
        )
        print("CLAUDE MODEL CALLS: 0")
        print("EXTERNAL ACTIONS EXECUTED: 0")
        print(
            "MCP INTEGRATION VERIFIED:",
            verified,
        )

        if not verified:
            raise RuntimeError(
                "Lab 7 validation failed. "
                "Paste the complete output."
            )

        print("V2 LAB 7 STATUS: COMPLETE")


if __name__ == "__main__":
    anyio.run(main)
