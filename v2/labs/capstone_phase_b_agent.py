"""Controlled premium-reasoning MCP agent for CASE-001."""

import json
import os
from pathlib import Path
import re
import time
from typing import Any

import anyio
from anthropic import Anthropic
from dotenv import load_dotenv
from mcp import Client
from mcp.types import TextContent

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
    TOOL_ARGUMENTS,
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
    / "capstone_phase_b_recommendation.json"
)

EXPECTED_CATALOG_ENTRY = (
    "haiku_45_premium_capstone"
)

DATA_CLASSIFICATION = "SYNTHETIC_TRAINING"

ALLOWED_RECOMMENDATIONS = {
    "APPROVE",
    "DECLINE",
    "REFER",
}


def extract_payload(
    result: Any,
) -> dict[str, Any]:
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


def extract_model_text(response: Any) -> str:
    return "\n".join(
        block.text
        for block in response.content
        if block.type == "text"
    ).strip()


def normalize_arguments(
    tool_name: str,
    arguments: Any,
) -> dict[str, Any]:
    if tool_name not in TOOL_ARGUMENTS:
        raise ValueError(
            "Tool is not registered."
        )

    if not isinstance(arguments, dict):
        raise ValueError(
            "Tool arguments must be an object."
        )

    expected = set(
        TOOL_ARGUMENTS[tool_name]
    )

    if set(arguments) != expected:
        raise ValueError(
            "Tool arguments do not match "
            "the approved schema."
        )

    normalized = {}

    for field in expected:
        value = arguments[field]

        if not isinstance(value, str):
            raise ValueError(
                f"{field} must be a string."
            )

        value = value.strip()

        if not value:
            raise ValueError(
                f"{field} cannot be blank."
            )

        if field in {"case_id", "location"}:
            value = value.upper()

        normalized[field] = value

    return normalized


def catalog_rate(
    entry: dict[str, Any],
    *possible_names: str,
) -> float:
    for name in possible_names:
        value = entry.get(name)

        if isinstance(value, (int, float)):
            return float(value)

    raise ValueError(
        "Verified catalog pricing is unavailable."
    )


def extract_recommendation(
    response_text: str,
) -> str | None:
    normalized = response_text.replace("*", "")

    match = re.search(
        r"AI_RECOMMENDATION\s*:\s*"
        r"(APPROVE|DECLINE|REFER)\b",
        normalized.upper(),
    )

    return match.group(1) if match else None


async def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")

    api_key = (
        os.getenv("ANTHROPIC_API_KEY")
        or ""
    ).strip()

    if not api_key:
        raise SystemExit(
            "ANTHROPIC_API_KEY is unavailable."
        )

    decision = route_request(
        task_type="UNDERWRITING_RECOMMENDATION",
        deterministic_answer=None,
        complexity="HIGH",
        business_risk="HIGH",
    )

    if decision.get("route") != "PREMIUM_REASONING":
        raise RuntimeError(
            "Expected PREMIUM_REASONING route."
        )

    if not is_data_classification_allowed(
        decision["route"],
        DATA_CLASSIFICATION,
    ):
        raise RuntimeError(
            "Data-classification policy blocked "
            "the selected route."
        )

    policy = get_route_policy(
        decision["route"]
    )

    catalog = load_model_catalog()

    selection = select_model(
        decision["route"],
        catalog,
        expected_input_tokens=3000,
        expected_output_tokens=500,
        consequential_action_requested=False,
    )

    if not all(
        (
            selection.get("selection_status")
            == "SELECTED",
            selection.get("catalog_entry_id")
            == EXPECTED_CATALOG_ENTRY,
            selection.get(
                "allowed_for_"
                "consequential_decisions"
            )
            is False,
        )
    ):
        raise RuntimeError(
            "No valid scoped premium model "
            "was selected."
        )

    model_id = selection["model_id"]

    selected_entry = next(
        (
            entry
            for entry in catalog
            if entry.get("catalog_entry_id")
            == selection["catalog_entry_id"]
        ),
        None,
    )

    if not isinstance(selected_entry, dict):
        raise RuntimeError(
            "Selected catalog entry is missing."
        )

    input_rate = catalog_rate(
        selected_entry,
        "input_cost_usd_per_million",
        "input_cost_per_million",
    )
    output_rate = catalog_rate(
        selected_entry,
        "output_cost_usd_per_million",
        "output_cost_per_million",
    )

    cost_limit = float(
        policy[
            "max_estimated_model_cost_usd"
        ]
    )
    policy_step_limit = int(
        policy["max_agent_steps"]
    )

    # A tighter cap for this exact task.
    execution_call_limit = min(
        3,
        policy_step_limit,
    )

    allowed_tools = frozenset(
        policy["allowed_tools"]
    )
    tool_definitions = (
        build_claude_tool_definitions(
            allowed_tools
        )
    )

    system_instructions = """
You are a controlled underwriting recommendation agent operating only
on synthetic training data.

You have exactly five read-only tools. You must use every tool exactly
once before producing the final response.

Efficiency sequence:
1. First call submission for CASE-001.
2. Then call weather_risk, compliance_rules, loss_history, and
   base_score together in the next response. Use the location returned
   by submission.
3. Then produce the final response.

Use only facts returned by the tools. Do not invent policy thresholds,
coverage terms, premiums, external definitions, or missing facts.
The base score is illustrative and has no approval threshold.

You may recommend APPROVE, DECLINE, or REFER, but you cannot authorize,
execute, price, bind, approve, or decline coverage.

Your final response must use these exact labels:

CASE_ID:
EVIDENCE_STATUS:
RISK_FACTORS:
LOSS_HISTORY:
BASE_SCORE:
POLICY_NEXT_STEP:
AI_RECOMMENDATION:
RATIONALE:
AUTHORIZATION_BOUNDARY:

AI_RECOMMENDATION must be exactly APPROVE, DECLINE, or REFER.
AUTHORIZATION_BOUNDARY must be HUMAN_DECISION_REQUIRED.
Keep the final response concise.
""".strip()

    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": (
                "Investigate synthetic underwriting "
                "case CASE-001 using every approved "
                "tool. Produce a recommendation only. "
                "Do not execute a business decision."
            ),
        }
    ]

    claude_client = Anthropic(
        api_key=api_key
    )

    tools_used: list[str] = []
    tool_payloads: dict[str, dict[str, Any]] = {}

    api_calls = 0
    tool_requests = 0
    mcp_tool_calls = 0
    rejected_tool_calls = 0
    total_input_tokens = 0
    total_output_tokens = 0
    total_model_latency_ms = 0.0
    total_mcp_latency_ms = 0.0
    final_text = ""
    final_stop_reason = None
    run_status = "STEP_LIMIT_REACHED"

    async with Client(
        mcp,
        raise_exceptions=True,
    ) as mcp_client:
        listed = await mcp_client.list_tools()

        server_tools = {
            tool.name
            for tool in listed.tools
        }

        if server_tools != allowed_tools:
            raise RuntimeError(
                "MCP tools do not match the "
                "control-plane allowlist."
            )

        for step in range(
            1,
            execution_call_limit + 1,
        ):
            print(
                f"\n--- CAPSTONE AGENT STEP "
                f"{step}/{execution_call_limit} ---"
            )

            model_started = time.perf_counter()

            response = (
                claude_client.messages.create(
                    model=model_id,
                    max_tokens=500,
                    system=system_instructions,
                    tools=tool_definitions,
                    messages=messages,
                )
            )

            model_latency_ms = (
                time.perf_counter()
                - model_started
            ) * 1000

            api_calls += 1
            total_model_latency_ms += (
                model_latency_ms
            )
            total_input_tokens += (
                response.usage.input_tokens
            )
            total_output_tokens += (
                response.usage.output_tokens
            )
            final_stop_reason = (
                response.stop_reason
            )

            actual_cost = (
                (
                    total_input_tokens
                    * input_rate
                )
                + (
                    total_output_tokens
                    * output_rate
                )
            ) / 1_000_000

            print(
                "INPUT TOKENS:",
                response.usage.input_tokens,
            )
            print(
                "OUTPUT TOKENS:",
                response.usage.output_tokens,
            )
            print(
                "STOP REASON:",
                response.stop_reason,
            )
            print(
                "CUMULATIVE MODEL COST USD:",
                f"{actual_cost:.8f}",
            )

            model_text = extract_model_text(
                response
            )

            if model_text:
                print("\nMODEL TEXT:")
                print(model_text)

            if actual_cost > cost_limit:
                run_status = "COST_LIMIT_EXCEEDED"
                break

            if response.stop_reason == "end_turn":
                final_text = model_text
                run_status = "MODEL_FINISHED"
                break

            if response.stop_reason != "tool_use":
                run_status = (
                    "UNSUPPORTED_STOP_REASON"
                )
                break

            messages.append(
                {
                    "role": "assistant",
                    "content": response.content,
                }
            )

            tool_results = []

            for block in response.content:
                if block.type != "tool_use":
                    continue

                tool_requests += 1
                tool_name = block.name
                supplied_arguments = block.input

                print(
                    "\nMODEL REQUESTED:",
                    tool_name,
                    json.dumps(
                        supplied_arguments
                    ),
                )

                try:
                    if tool_name not in allowed_tools:
                        raise ValueError(
                            "Tool is not allowed."
                        )

                    if tool_name in tools_used:
                        raise ValueError(
                            "Duplicate tool request."
                        )

                    normalized_arguments = (
                        normalize_arguments(
                            tool_name,
                            supplied_arguments,
                        )
                    )

                    mcp_started = (
                        time.perf_counter()
                    )

                    mcp_result = (
                        await mcp_client.call_tool(
                            tool_name,
                            normalized_arguments,
                        )
                    )

                    mcp_latency_ms = (
                        time.perf_counter()
                        - mcp_started
                    ) * 1000

                    total_mcp_latency_ms += (
                        mcp_latency_ms
                    )

                    if getattr(
                        mcp_result,
                        "is_error",
                        True,
                    ):
                        raise ValueError(
                            "MCP tool returned an error."
                        )

                    payload = extract_payload(
                        mcp_result
                    )

                    tools_used.append(tool_name)
                    tool_payloads[tool_name] = (
                        payload
                    )
                    mcp_tool_calls += 1

                    print("APPLICATION GATE: ALLOWED")
                    print("MCP RESULT:")
                    print(
                        json.dumps(
                            payload,
                            indent=2,
                        )
                    )

                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": json.dumps(
                                payload
                            ),
                        }
                    )

                except (
                    TypeError,
                    ValueError,
                    KeyError,
                ) as error:
                    rejected_tool_calls += 1

                    print(
                        "APPLICATION GATE: BLOCKED"
                    )
                    print(
                        "REASON:",
                        type(error).__name__,
                    )

                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": json.dumps(
                                {
                                    "error": (
                                        "Governed tool "
                                        "request rejected"
                                    )
                                }
                            ),
                            "is_error": True,
                        }
                    )

            if not tool_results:
                run_status = (
                    "EMPTY_TOOL_REQUEST"
                )
                break

            messages.append(
                {
                    "role": "user",
                    "content": tool_results,
                }
            )

    actual_cost = (
        (
            total_input_tokens * input_rate
        )
        + (
            total_output_tokens * output_rate
        )
    ) / 1_000_000

    submission_payload = tool_payloads.get(
        "submission",
        {},
    )
    compliance_payload = tool_payloads.get(
        "compliance_rules",
        {},
    )
    loss_payload = tool_payloads.get(
        "loss_history",
        {},
    )
    weather_payload = tool_payloads.get(
        "weather_risk",
        {},
    )
    score_payload = tool_payloads.get(
        "base_score",
        {},
    )

    tools_valid = all(
        (
            set(tools_used) == allowed_tools,
            len(tools_used)
            == len(allowed_tools),
            mcp_tool_calls
            == len(allowed_tools),
            rejected_tool_calls == 0,
        )
    )

    authoritative_facts_valid = all(
        (
            submission_payload.get(
                "evidence_status"
            )
            == "COMPLETE",
            submission_payload.get(
                "outstanding_evidence"
            )
            == [],
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
            len(
                loss_payload.get(
                    "records",
                    [],
                )
            )
            == 2,
            weather_payload.get(
                "coastal_exposure"
            )
            == "HIGH",
            weather_payload.get(
                "hurricane_exposure"
            )
            == "HIGH",
            weather_payload.get(
                "flood_zone"
            )
            == "AE",
            score_payload.get("score") == 75,
        )
    )

    recommendation = extract_recommendation(
        final_text
    )
    normalized_final = final_text.replace(
        "*",
        "",
    ).upper()

    output_contract_valid = all(
        (
            run_status == "MODEL_FINISHED",
            final_stop_reason == "end_turn",
            recommendation
            in ALLOWED_RECOMMENDATIONS,
            "CASE_ID:" in normalized_final,
            "CASE-001" in normalized_final,
            "EVIDENCE_STATUS:" in normalized_final,
            "COMPLETE" in normalized_final,
            "POLICY_NEXT_STEP:"
            in normalized_final,
            "HUMAN_REVIEW"
            in normalized_final,
            "AUTHORIZATION_BOUNDARY:"
            in normalized_final,
            "HUMAN_DECISION_REQUIRED"
            in normalized_final,
        )
    )

    cost_valid = all(
        (
            actual_cost <= cost_limit,
            api_calls
            <= execution_call_limit,
            api_calls <= policy_step_limit,
        )
    )

    phase_b_verified = all(
        (
            selection.get(
                "catalog_entry_id"
            )
            == EXPECTED_CATALOG_ENTRY,
            selection.get(
                "allowed_for_"
                "consequential_decisions"
            )
            is False,
            tools_valid,
            authoritative_facts_valid,
            output_contract_valid,
            cost_valid,
        )
    )

    report = {
        "case_id": "CASE-001",
        "route_decision": decision,
        "model_selection": selection,
        "data_classification": (
            DATA_CLASSIFICATION
        ),
        "policy_max_agent_steps": (
            policy_step_limit
        ),
        "execution_call_limit": (
            execution_call_limit
        ),
        "tools_used": tools_used,
        "tool_payloads": tool_payloads,
        "recommendation": recommendation,
        "recommendation_text": final_text,
        "authorization_status": (
            "PENDING_HUMAN_DECISION"
        ),
        "metrics": {
            "claude_api_calls": api_calls,
            "mcp_tool_requests": tool_requests,
            "mcp_tool_calls": mcp_tool_calls,
            "rejected_tool_calls": (
                rejected_tool_calls
            ),
            "input_tokens": (
                total_input_tokens
            ),
            "output_tokens": (
                total_output_tokens
            ),
            "model_latency_ms": round(
                total_model_latency_ms,
                2,
            ),
            "mcp_latency_ms": round(
                total_mcp_latency_ms,
                2,
            ),
            "input_rate_per_million": (
                input_rate
            ),
            "output_rate_per_million": (
                output_rate
            ),
            "actual_model_cost_usd": round(
                actual_cost,
                8,
            ),
            "cost_limit_usd": cost_limit,
            "stop_reason": (
                final_stop_reason
            ),
        },
        "validation": {
            "tools_valid": tools_valid,
            "authoritative_facts_valid": (
                authoritative_facts_valid
            ),
            "output_contract_valid": (
                output_contract_valid
            ),
            "cost_valid": cost_valid,
            "phase_b_verified": (
                phase_b_verified
            ),
        },
        "external_action_executed": False,
        "coverage_decision_executed": False,
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
                "PREMIUM_MCP_RECOMMENDATION"
            ),
            "case_id": "CASE-001",
            "route": decision.get("route"),
            "catalog_entry_id": (
                selection.get(
                    "catalog_entry_id"
                )
            ),
            "model_id": model_id,
            "tools_used": tools_used,
            "model_calls": api_calls,
            "mcp_tool_calls": mcp_tool_calls,
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "estimated_model_cost_usd": (
                selection.get(
                    "estimated_model_cost_usd"
                )
            ),
            "actual_model_cost_usd": round(
                actual_cost,
                8,
            ),
            "latency_ms": round(
                total_model_latency_ms,
                2,
            ),
            "recommendation": recommendation,
            "authorization_status": (
                "PENDING_HUMAN_DECISION"
            ),
            "external_action_executed": False,
            "coverage_decision_executed": False,
            "verified": phase_b_verified,
        }
    )

    print("\nFINAL AI RECOMMENDATION:")
    print(final_text)

    print("\nPHASE B METRICS:")
    print(
        json.dumps(
            report["metrics"],
            indent=2,
        )
    )

    print(
        "\nTOOLS USED EXACTLY ONCE:",
        tools_valid,
    )
    print(
        "AUTHORITATIVE FACTS VALID:",
        authoritative_facts_valid,
    )
    print(
        "OUTPUT CONTRACT VALID:",
        output_contract_valid,
    )
    print(
        "COST LIMIT VALID:",
        cost_valid,
    )
    print(
        "AI RECOMMENDATION:",
        recommendation,
    )
    print(
        "AUTHORIZATION STATUS: "
        "PENDING_HUMAN_DECISION"
    )
    print(
        "PHASE B VERIFIED:",
        phase_b_verified,
    )
    print("EXTERNAL ACTIONS EXECUTED: 0")
    print("COVERAGE DECISIONS EXECUTED: 0")
    print("SECRETS PRINTED: False")

    if not phase_b_verified:
        raise RuntimeError(
            "Capstone Phase B validation failed. "
            "No business action was executed."
        )

    print(
        "CAPSTONE PHASE B RECOMMENDATION: "
        "COMPLETE"
    )


if __name__ == "__main__":
    anyio.run(main)
