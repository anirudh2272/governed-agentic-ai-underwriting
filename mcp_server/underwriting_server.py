from mcp.server import MCPServer

from tools.underwriting_tools import (
    submission as lookup_submission,
    weather_risk as lookup_weather_risk,
    compliance_rules as lookup_compliance_rules,
    loss_history as lookup_loss_history,
    base_score as calculate_base_score,
)

mcp = MCPServer("underwriting-training-server")


@mcp.tool()
def submission(case_id: str) -> dict:
    """Retrieve a synthetic underwriting submission and evidence status.

    Args:
        case_id: Synthetic underwriting case identifier.
    """
    return lookup_submission(case_id)


@mcp.tool()
def weather_risk(location: str) -> dict:
    """Retrieve synthetic environmental risk for a location.

    Args:
        location: Location identifier returned by the submission tool.
    """
    return lookup_weather_risk(location)


@mcp.tool()
def compliance_rules(case_id: str) -> dict:
    """Retrieve evidence, review, and approval requirements.

    Args:
        case_id: Synthetic underwriting case identifier.
    """
    return lookup_compliance_rules(case_id)


@mcp.tool()
def loss_history(case_id: str) -> dict:
    """Retrieve contractor loss-history evidence status.

    Missing evidence must not be interpreted as zero losses.

    Args:
        case_id: Synthetic underwriting case identifier.
    """
    return lookup_loss_history(case_id)


@mcp.tool()
def base_score(case_id: str) -> dict:
    """Calculate the illustrative deterministic training score.

    The result is not a real insurance risk score.

    Args:
        case_id: Synthetic underwriting case identifier.
    """
    return calculate_base_score(case_id)


if __name__ == "__main__":
    # Standard input/output carries MCP protocol messages.
    # Do not add print statements while this server is running.
    mcp.run(transport="stdio")
