import json
import sys
from pathlib import Path

import anyio
from mcp import Client, StdioServerParameters
from mcp.types import TextContent

PROJECT_ROOT = Path(__file__).resolve().parent.parent

EXPECTED_TOOLS = {
    "submission",
    "weather_risk",
    "compliance_rules",
    "loss_history",
    "base_score",
}


async def main():
    server = StdioServerParameters(
        command=sys.executable,
        args=[
            "-m",
            "mcp_server.underwriting_server",
        ],
        env={
            "PYTHONPATH": str(PROJECT_ROOT),
        },
    )

    print("STARTING MCP SERVER...")

    async with Client(server) as client:
        print("CONNECTION: SUCCESS")
        print("PROTOCOL VERSION:", client.protocol_version)

        listed = await client.list_tools()
        tool_names = {
            tool.name
            for tool in listed.tools
        }

        print("\nDISCOVERED TOOLS:")

        for tool in sorted(
            listed.tools,
            key=lambda item: item.name,
        ):
            print(f"- {tool.name}")

        missing = EXPECTED_TOOLS - tool_names
        unexpected = tool_names - EXPECTED_TOOLS

        if missing or unexpected:
            raise RuntimeError(
                f"Tool mismatch. Missing={sorted(missing)}, "
                f"Unexpected={sorted(unexpected)}"
            )

        print("\nTOOL DISCOVERY: VERIFIED")

        result = await client.call_tool(
            "submission",
            {"case_id": "CASE-001"},
        )

        print(
            "\nSUBMISSION CALL ERROR:",
            result.is_error,
        )

        if result.is_error:
            raise RuntimeError(
                "MCP submission call returned an error."
            )

        text_results = [
            block.text
            for block in result.content
            if isinstance(block, TextContent)
        ]

        if not text_results:
            raise RuntimeError(
                "MCP submission returned no text content."
            )

        print("SUBMISSION MODEL-FACING CONTENT:")

        for text in text_results:
            print(text)

        submission_result = json.loads(
            text_results[0]
        )

        if (
            submission_result.get("case_id")
            != "CASE-001"
        ):
            raise RuntimeError(
                "Unexpected case returned by MCP."
            )

        if (
            submission_result.get("evidence_status")
            != "INCOMPLETE"
        ):
            raise RuntimeError(
                "Unexpected evidence status."
            )

        print(
            "\nSTRUCTURED CONTENT AVAILABLE:",
            result.structured_content is not None,
        )
        print("MCP RESULT VALIDATION: SUCCESS")
        print("MCP TOOL CALL: SUCCESS")


if __name__ == "__main__":
    anyio.run(main)
