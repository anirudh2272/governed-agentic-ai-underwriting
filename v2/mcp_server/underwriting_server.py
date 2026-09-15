from typing import Any

from mcp.server import MCPServer

from v2.control_plane.tool_gateway import (
    DEFAULT_ALLOWED_TOOLS,
    execute_governed_tool,
)

mcp = MCPServer("v2-governed-underwriting")


def _execute(
    tool_name: str,
    arguments: dict[str, Any],
) -> dict:
    envelope = execute_governed_tool(
        tool_name,
        arguments,
        allowed_tools=DEFAULT_ALLOWED_TOOLS,
    )

    if envelope["status"] != "SUCCESS":
        return envelope

    return envelope["result"]


@mcp.tool()
def submission(case_id: str) -> dict:
    """Retrieve an authoritative synthetic case."""
    return _execute(
        "submission",
        {"case_id": case_id},
    )


@mcp.tool()
def weather_risk(location: str) -> dict:
    """Retrieve synthetic weather-risk data."""
    return _execute(
        "weather_risk",
        {"location": location},
    )


@mcp.tool()
def compliance_rules(case_id: str) -> dict:
    """Retrieve authoritative compliance policy."""
    return _execute(
        "compliance_rules",
        {"case_id": case_id},
    )


@mcp.tool()
def loss_history(case_id: str) -> dict:
    """Retrieve synthetic loss-history evidence."""
    return _execute(
        "loss_history",
        {"case_id": case_id},
    )


@mcp.tool()
def base_score(case_id: str) -> dict:
    """Calculate the illustrative training score."""
    return _execute(
        "base_score",
        {"case_id": case_id},
    )


if __name__ == "__main__":
    # MCP protocol messages use standard input/output.
    # Do not add print statements here.
    mcp.run(transport="stdio")
