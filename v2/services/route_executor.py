from time import perf_counter
from typing import Any


MODEL_ROUTES = {
    "CONVERSATIONAL",
    "PREMIUM_REASONING",
}


def execute_route(
    decision: dict[str, Any],
    *,
    client: Any = None,
    model_id: str | None = None,
    system: str | None = None,
    messages: list[dict[str, Any]] | None = None,
    max_tokens: int = 300,
) -> dict[str, Any]:
    """Execute the route selected by the control plane."""

    route = decision.get("route")

    if route == "NO_LLM":
        started = perf_counter()

        result = {
            "route": route,
            "requested_capability": None,
            "execution_mode": "DETERMINISTIC",
            "model_id": None,
            "model_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "response_text": (
                decision["deterministic_result"]
            ),
            "stop_reason": (
                "DETERMINISTIC_COMPLETION"
            ),
            "latency_ms": round(
                (perf_counter() - started) * 1000,
                2,
            ),
        }

        return result

    if route not in MODEL_ROUTES:
        raise ValueError(
            f"Unsupported execution route: {route}"
        )

    if decision.get("model_required") is not True:
        raise ValueError(
            "Model route must set model_required=True."
        )

    expected_capability = route

    if (
        decision.get("requested_capability")
        != expected_capability
    ):
        raise ValueError(
            "Route and requested capability disagree."
        )

    if client is None:
        raise ValueError(
            "A Claude client is required."
        )

    if not model_id:
        raise ValueError("model_id is required.")

    if not system:
        raise ValueError("system is required.")

    if not messages:
        raise ValueError("messages are required.")

    if max_tokens <= 0:
        raise ValueError(
            "max_tokens must be greater than zero."
        )

    started = perf_counter()

    response = client.messages.create(
        model=model_id,
        max_tokens=max_tokens,
        system=system,
        messages=messages,
    )

    latency_ms = round(
        (perf_counter() - started) * 1000,
        2,
    )

    response_text = "\n".join(
        block.text
        for block in response.content
        if block.type == "text"
    ).strip()

    if not response_text:
        raise RuntimeError("Claude returned no text.")

    usage = response.usage

    return {
        "route": route,
        "requested_capability": (
            decision["requested_capability"]
        ),
        "execution_mode": "MODEL",
        "model_id": response.model,
        "model_calls": 1,
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cache_creation_input_tokens": (
            getattr(
                usage,
                "cache_creation_input_tokens",
                0,
            )
            or 0
        ),
        "cache_read_input_tokens": (
            getattr(
                usage,
                "cache_read_input_tokens",
                0,
            )
            or 0
        ),
        "response_text": response_text,
        "stop_reason": response.stop_reason,
        "latency_ms": latency_ms,
    }
