import json
import os
import re
import time
from pathlib import Path
from typing import Any

import anyio
from anthropic import Anthropic
from dotenv import load_dotenv
from mcp import Client
from mcp.types import TextContent

from tools.underwriting_controls import (
    authorize_underwriting_step,
)
from v2.control_plane.tool_gateway import (
    DEFAULT_ALLOWED_TOOLS,
)
from v2.control_plane.tool_schemas import (
    build_claude_tool_definitions,
)
from v2.mcp_server.underwriting_server import mcp
from v2.services.telemetry import record_event


PROJECT_ROOT = Path(__file__).resolve().parents[2]

REQUIRED_TOOLS = {
    "submission",
    "compliance_rules",
}

FORBIDDEN_MODEL_TOOLS = {
    "authorize_underwriting_step",
    "bind_coverage",
    "approve",
    "decline",
    "set_premium",
}


def extract_mcp_payload(
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

        if isinstance(candidate, dict):
            payload = candidate.get(
                "result",
                candidate,
            )

            if isinstance(payload, dict):
                return payload

    raise RuntimeError(
        "MCP returned no usable dictionary payload."
    )


def extract_mcp_error(result: Any) -> str:
    messages = []

    for block in result.content:
        if isinstance(block, TextContent):
            messages.append(block.text)

    return "\n".join(messages) or "Unknown MCP error"


def extract_model_text(response: Any) -> str:
    return "\n".join(
        block.text
        for block in response.content
        if block.type == "text"
        and block.text.strip()
    )


def extract_recommended_action(
    response_text: str,
) -> str:
    normalized = response_text.upper().replace(
        "*",
        "",
    )

    match = re.search(
        r"RECOMMENDED_NEXT_STEP\s*:\s*"
        r"([A-Z][A-Z0-9_]*)",
        normalized,
    )

    return match.group(1) if match else "UNKNOWN"


async def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")

    api_key = os.getenv(
        "ANTHROPIC_API_KEY",
        "",
    ).strip()
    model = os.getenv(
        "CLAUDE_MODEL",
        "",
    ).strip()

    if not api_key:
        raise SystemExit(
            "ANTHROPIC_API_KEY is missing."
        )

    if not model:
        raise SystemExit(
            "CLAUDE_MODEL is missing."
        )

    claude = Anthropic(api_key=api_key)

    system_instructions = """
You are a controlled underwriting workflow assistant
working only with synthetic training data.

You must call submission and compliance_rules before
giving a recommendation. Use other available tools only
when necessary.

Treat MCP tool results as the authoritative facts.
Do not invent missing facts, policy meanings, evidence
periods, pricing, approval, decline, or binding decisions.

You may recommend only the exact next_step returned by
compliance_rules. You cannot execute external actions or
coverage decisions.

Return these exact sections:
CASE_FACTS
EVIDENCE_GAPS
POLICY_NEXT_STEP
AUTHORIZATION_BOUNDARY

End with one plain machine-readable line:
RECOMMENDED_NEXT_STEP: <exact policy next_step>
""".strip()

    task = (
        "Investigate synthetic underwriting case "
        "CASE-001 and recommend only its currently "
        "permitted next workflow step."
    )

    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": task,
        }
    ]

    api_calls = 0
    mcp_tool_calls = 0
    rejected_tool_calls = 0
    total_input_tokens = 0
    total_output_tokens = 0
    total_model_latency_ms = 0.0
    total_mcp_latency_ms = 0.0
    final_text = ""
    final_stop_reason = "NOT_STARTED"
    used_tools: list[str] = []
    captured_payloads: dict[
        str,
        dict[str, Any],
    ] = {}

    async with Client(
        mcp,
        raise_exceptions=True,
    ) as mcp_client:
        listed = await mcp_client.list_tools()
        server_tools = {
            tool.name
            for tool in listed.tools
        }

        local_schema_tools = {
            definition["name"]
            for definition
            in build_claude_tool_definitions()
        }

        allowed_tools = set(
            DEFAULT_ALLOWED_TOOLS
        )

        tool_contracts_match = (
            server_tools
            == local_schema_tools
            == allowed_tools
        )

        forbidden_exposed = (
            server_tools.intersection(
                FORBIDDEN_MODEL_TOOLS
            )
        )

        print("MCP CONNECTION: SUCCESS")
        print(
            "MCP PROTOCOL VERSION:",
            mcp_client.protocol_version,
        )
        print(
            "MCP SERVER TOOLS:",
            sorted(server_tools),
        )
        print(
            "TOOL CONTRACTS MATCH:",
            tool_contracts_match,
        )
        print(
            "FORBIDDEN TOOLS EXPOSED:",
            sorted(forbidden_exposed),
        )

        if (
            not tool_contracts_match
            or forbidden_exposed
        ):
            raise RuntimeError(
                "MCP tool exposure does not match "
                "the approved application contract."
            )

        claude_tools = [
            {
                "name": tool.name,
                "description": (
                    tool.description
                    or f"Governed MCP tool: {tool.name}"
                ),
                "input_schema": tool.input_schema,
            }
            for tool in sorted(
                listed.tools,
                key=lambda item: item.name,
            )
        ]

        for step in range(1, 7):
            print(
                f"\n--- CONTROLLED MCP AGENT "
                f"STEP {step}/6 ---"
            )

            model_started = time.perf_counter()

            response = claude.messages.create(
                model=model,
                max_tokens=1000,
                system=system_instructions,
                tools=claude_tools,
                messages=messages,
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
            final_stop_reason = str(
                response.stop_reason
            )

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

            model_text = extract_model_text(
                response
            )

            if model_text:
                print("\nMODEL TEXT:")
                print(model_text)

            if response.stop_reason == "end_turn":
                final_text = model_text
                break

            if response.stop_reason != "tool_use":
                raise RuntimeError(
                    "Unexpected Claude stop reason: "
                    f"{response.stop_reason}"
                )

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

                tool_name = block.name
                arguments = dict(block.input)

                print(
                    "\nCLAUDE REQUESTED:",
                    tool_name,
                    json.dumps(arguments),
                )

                tool_allowed = (
                    tool_name in allowed_tools
                    and tool_name in server_tools
                )

                print(
                    "APPLICATION TOOL GATE:",
                    (
                        "ALLOWED"
                        if tool_allowed
                        else "BLOCKED"
                    ),
                )

                if not tool_allowed:
                    rejected_tool_calls += 1

                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": (
                                "Tool request blocked by "
                                "application allowlist."
                            ),
                            "is_error": True,
                        }
                    )
                    continue

                mcp_started = time.perf_counter()

                mcp_result = (
                    await mcp_client.call_tool(
                        tool_name,
                        arguments,
                    )
                )

                total_mcp_latency_ms += (
                    time.perf_counter()
                    - mcp_started
                ) * 1000
                mcp_tool_calls += 1
                used_tools.append(tool_name)

                if mcp_result.is_error:
                    result_content = (
                        extract_mcp_error(
                            mcp_result
                        )
                    )
                    print("MCP RESULT: ERROR")
                else:
                    payload = extract_mcp_payload(
                        mcp_result
                    )
                    captured_payloads[
                        tool_name
                    ] = payload
                    result_content = json.dumps(
                        payload
                    )
                    print("MCP RESULT: SUCCESS")

                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result_content,
                        "is_error": bool(
                            mcp_result.is_error
                        ),
                    }
                )

            if not tool_results:
                raise RuntimeError(
                    "Claude stopped for tool use but "
                    "supplied no tool request."
                )

            messages.append(
                {
                    "role": "user",
                    "content": tool_results,
                }
            )
        else:
            raise RuntimeError(
                "Agent exceeded the six-step limit."
            )

    required_tools_used = (
        REQUIRED_TOOLS.issubset(
            set(used_tools)
        )
    )

    compliance = captured_payloads.get(
        "compliance_rules"
    )

    if not compliance:
        raise RuntimeError(
            "The agent did not retrieve compliance "
            "rules through MCP."
        )

    recommended_action = (
        extract_recommended_action(final_text)
    )
    policy_next_step = str(
        compliance.get("next_step", "")
    ).upper()

    authorization = (
        authorize_underwriting_step(
            case_id="CASE-001",
            requested_action=(
                recommended_action
            ),
            compliance=compliance,
            human_approval_supplied=False,
        )
    )

    boundary_test = (
        authorize_underwriting_step(
            case_id="CASE-001",
            requested_action="BIND_COVERAGE",
            compliance=compliance,
            human_approval_supplied=False,
        )
    )

    lab_verified = all(
        (
            bool(final_text),
            final_stop_reason == "end_turn",
            tool_contracts_match,
            not forbidden_exposed,
            required_tools_used,
            rejected_tool_calls == 0,
            (
                recommended_action
                == policy_next_step
                == "REQUEST_EVIDENCE"
            ),
            (
                authorization.get(
                    "authorization_status"
                )
                == "PERMITTED"
            ),
            (
                authorization.get(
                    "external_action_executed"
                )
                is False
            ),
            (
                authorization.get(
                    "coverage_decision_executed"
                )
                is False
            ),
            (
                boundary_test.get(
                    "authorization_status"
                )
                == "BLOCKED"
            ),
            (
                boundary_test.get(
                    "coverage_decision_executed"
                )
                is False
            ),
        )
    )

    telemetry_event = {
        "lab": "V2_LAB_08",
        "route": "CONTROLLED_MCP_AGENT",
        "model_id": model,
        "context_strategy": "MCP_TOOL_DISCOVERY",
        "context_characters": (
            len(system_instructions)
            + len(task)
        ),
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "latency_ms": round(
            total_model_latency_ms,
            2,
        ),
        "mcp_latency_ms": round(
            total_mcp_latency_ms,
            2,
        ),
        "model_calls": api_calls,
        "tool_calls": mcp_tool_calls,
        "mcp_tool_calls": mcp_tool_calls,
        "rejected_tool_calls": (
            rejected_tool_calls
        ),
        "tools_used": used_tools,
        "stop_reason": final_stop_reason,
        "recommended_action": (
            recommended_action
        ),
        "policy_next_step": policy_next_step,
        "authorization_status": (
            authorization.get(
                "authorization_status"
            )
        ),
        "external_action_executed": (
            authorization.get(
                "external_action_executed"
            )
        ),
        "coverage_decision_executed": (
            authorization.get(
                "coverage_decision_executed"
            )
        ),
        "lab_verified": lab_verified,
    }

    record_event(telemetry_event)

    summary = {
        "claude_api_calls": api_calls,
        "mcp_tool_calls": mcp_tool_calls,
        "tools_used": used_tools,
        "required_tools_used": (
            required_tools_used
        ),
        "rejected_tool_calls": (
            rejected_tool_calls
        ),
        "recommended_action": (
            recommended_action
        ),
        "policy_next_step": policy_next_step,
        "model_matches_policy": (
            recommended_action
            == policy_next_step
        ),
        "authorization_status": (
            authorization.get(
                "authorization_status"
            )
        ),
        "external_action_executed": (
            authorization.get(
                "external_action_executed"
            )
        ),
        "coverage_decision_executed": (
            authorization.get(
                "coverage_decision_executed"
            )
        ),
        "bind_coverage_boundary": (
            boundary_test.get(
                "authorization_status"
            )
        ),
        "total_input_tokens": (
            total_input_tokens
        ),
        "total_output_tokens": (
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
        "lab_verified": lab_verified,
    }

    print("\nFINAL CONTROLLED RECOMMENDATION:")
    print(final_text)

    print("\nAPPLICATION AUTHORIZATION:")
    print(
        json.dumps(
            authorization,
            indent=2,
        )
    )

    print("\nBIND_COVERAGE BOUNDARY TEST:")
    print(
        json.dumps(
            boundary_test,
            indent=2,
        )
    )

    print("\nLAB 8 SUMMARY:")
    print(json.dumps(summary, indent=2))

    print("\nSECRETS PRINTED: False")

    if not lab_verified:
        raise RuntimeError(
            "Lab 8 validation failed. "
            "Paste the complete output."
        )

    print(
        "V2 LAB 8 STATUS: COMPLETE"
    )


if __name__ == "__main__":
    anyio.run(main)
