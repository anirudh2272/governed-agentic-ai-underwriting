from collections.abc import Iterable
from typing import Any

from v2.control_plane.tool_gateway import (
    DEFAULT_ALLOWED_TOOLS,
    TOOL_ARGUMENTS,
    TOOL_REGISTRY,
)

TOOL_DESCRIPTIONS = {
    "submission": (
        "Retrieve the authoritative synthetic "
        "underwriting case."
    ),
    "weather_risk": (
        "Retrieve synthetic weather risk for a location."
    ),
    "compliance_rules": (
        "Retrieve authoritative policy requirements "
        "and the next workflow step."
    ),
    "loss_history": (
        "Retrieve contractor loss-history evidence."
    ),
    "base_score": (
        "Calculate the illustrative deterministic "
        "training score."
    ),
}


def build_claude_tool_definitions(
    allowed_tools: Iterable[str] = (
        DEFAULT_ALLOWED_TOOLS
    ),
) -> list[dict[str, Any]]:
    allowed = set(allowed_tools)
    unknown = allowed - set(TOOL_REGISTRY)

    if unknown:
        raise ValueError(
            "Allowlist contains unregistered tools: "
            f"{sorted(unknown)}"
        )

    definitions = []

    for tool_name in sorted(allowed):
        argument_names = TOOL_ARGUMENTS[tool_name]

        definitions.append(
            {
                "name": tool_name,
                "description": (
                    TOOL_DESCRIPTIONS[tool_name]
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        argument_name: {
                            "type": "string",
                            "description": (
                                "Synthetic training "
                                "identifier."
                            ),
                        }
                        for argument_name
                        in argument_names
                    },
                    "required": sorted(argument_names),
                    "additionalProperties": False,
                },
            }
        )

    return definitions
