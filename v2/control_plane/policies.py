"""Central policies for the V2 training control plane.

These values are illustrative training controls. They are not
production insurance, privacy, or financial policies.
"""

from typing import Any

from v2.control_plane.tool_gateway import (
    DEFAULT_ALLOWED_TOOLS,
)


VALID_DATA_CLASSIFICATIONS = frozenset(
    {
        "PUBLIC",
        "INTERNAL",
        "CONFIDENTIAL",
        "RESTRICTED",
        "SYNTHETIC_TRAINING",
    }
)


ROUTE_POLICIES: dict[str, dict[str, Any]] = {
    "NO_LLM": {
        "allowed_data_classifications": (
            VALID_DATA_CLASSIFICATIONS
        ),
        "allowed_tools": frozenset(),
        "max_agent_steps": 0,
        "max_estimated_model_cost_usd": 0.0,
    },
    "CONVERSATIONAL": {
        "allowed_data_classifications": frozenset(
            {
                "PUBLIC",
                "INTERNAL",
                "SYNTHETIC_TRAINING",
            }
        ),
        "allowed_tools": frozenset(),
        "max_agent_steps": 1,
        "max_estimated_model_cost_usd": 0.003,
    },
    "PREMIUM_REASONING": {
        "allowed_data_classifications": frozenset(
            {
                "PUBLIC",
                "INTERNAL",
                "SYNTHETIC_TRAINING",
            }
        ),
        "allowed_tools": DEFAULT_ALLOWED_TOOLS,
        "max_agent_steps": 6,
        "max_estimated_model_cost_usd": 0.01,
    },
}


def get_route_policy(route: str) -> dict[str, Any]:
    """Return the governed policy for a logical route."""

    normalized_route = route.strip().upper()
    policy = ROUTE_POLICIES.get(normalized_route)

    if policy is None:
        raise ValueError(
            f"Unsupported control-plane route: "
            f"{normalized_route}"
        )

    return {
        "route": normalized_route,
        "allowed_data_classifications": frozenset(
            policy["allowed_data_classifications"]
        ),
        "allowed_tools": frozenset(
            policy["allowed_tools"]
        ),
        "max_agent_steps": int(
            policy["max_agent_steps"]
        ),
        "max_estimated_model_cost_usd": float(
            policy["max_estimated_model_cost_usd"]
        ),
    }


def is_data_classification_allowed(
    route: str,
    data_classification: str,
) -> bool:
    """Check whether a route may process a data class."""

    normalized_classification = (
        data_classification.strip().upper()
    )

    if normalized_classification not in (
        VALID_DATA_CLASSIFICATIONS
    ):
        return False

    policy = get_route_policy(route)

    return normalized_classification in policy[
        "allowed_data_classifications"
    ]
