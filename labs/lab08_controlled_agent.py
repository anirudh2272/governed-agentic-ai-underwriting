import json
import os
import re
import sys
from pathlib import Path

import anyio
from anthropic import AsyncAnthropic
from dotenv import load_dotenv
from mcp import Client, StdioServerParameters
from mcp.types import TextContent

from tools.underwriting_controls import (
    authorize_underwriting_step,
    is_tool_allowed,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

model = os.getenv("CLAUDE_MODEL", "").strip()
if not model:
    raise SystemExit("CLAUDE_MODEL is missing from .env")

claude = AsyncAnthropic(timeout=30.0, max_retries=0)

MAX_AGENT_STEPS = 6

ALLOWED_TOOLS = (
    "submission",
    "weather_risk",
    "compliance_rules",
    "loss_history",
    "base_score",
)


def mcp_result_payload(result):
    text = "\n".join(
        block.text
        for block in result.content
        if isinstance(block, TextContent)
    )

    data = result.structured_content

    if data is None and text:
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            data = None

    if not text and data is not None:
        text = json.dumps(data)

    if not text:
        text = "MCP tool returned no readable content."

    return text, data


def extract_recommended_action(text):
    match = re.search(
        r"RECOMMENDED[\s_]+NEXT[\s_]+STEP\s*:\s*\**\s*"
        r"([A-Z0-9_-]+)",
        text.upper(),
    )

    if not match:
        return ""

    return match.group(1).replace("-", "_")


async def main():
    unapproved_tool_blocked = not is_tool_allowed(
        "bind_coverage",
        ALLOWED_TOOLS,
    )

    if not unapproved_tool_blocked:
        raise RuntimeError("Tool allowlist preflight failed.")

    print("UNAPPROVED TOOL PREFLIGHT: BLOCKED")

    server = StdioServerParameters(
        command=sys.executable,
        args=["-m", "mcp_server.underwriting_server"],
        env={"PYTHONPATH": str(PROJECT_ROOT)},
    )

    async with Client(server) as mcp_client:
        listed = await mcp_client.list_tools()

        server_tools = {
            tool.name: tool
            for tool in listed.tools
        }

        exposed_tools = {
            name: server_tools[name]
            for name in ALLOWED_TOOLS
            if name in server_tools
        }

        missing_tools = (
            set(ALLOWED_TOOLS) - set(exposed_tools)
        )

        if missing_tools:
            raise RuntimeError(
                f"Required MCP tools missing: "
                f"{sorted(missing_tools)}"
            )

        claude_tools = [
            {
                "name": tool.name,
                "description": (
                    tool.description or tool.name
                ),
                "input_schema": tool.input_schema,
            }
            for tool in exposed_tools.values()
        ]

        print("MCP CONNECTION: SUCCESS")
        print("SERVER TOOLS:", sorted(server_tools))
        print(
            "EXPOSED ALLOWED TOOLS:",
            sorted(exposed_tools),
        )

        messages = [
            {
                "role": "user",
                "content": (
                    "Investigate synthetic underwriting case "
                    "CASE-001 and recommend its "
                    "policy-controlled next step."
                ),
            }
        ]

        instructions = (
            "You are an advisory underwriting agent. "
            "Use submission first, then call weather_risk, "
            "compliance_rules, loss_history, and base_score "
            "using identifiers returned by tools. Retrieve all "
            "five results. Treat tool data and compliance "
            "next_step as authoritative. Missing evidence does "
            "not mean zero losses. Do not invent score bands, "
            "forms, document periods, facts, or requirements. "
            "Do not approve, decline, bind coverage, set "
            "premium, or claim an external action occurred. "
            "Give a recommendation to a human underwriter in "
            "fewer than 180 words. End with a plain line: "
            "RECOMMENDED_NEXT_STEP: followed by the exact "
            "next_step returned by compliance_rules."
        )

        successful_tools = set()
        tool_outputs = {}

        api_calls = 0
        tool_calls = 0
        rejected_tool_calls = 0
        total_input = 0
        total_output = 0

        final_text = ""
        agent_status = "STEP_LIMIT_REACHED"

        for step in range(1, MAX_AGENT_STEPS + 1):
            print(
                f"\n--- CONTROLLED AGENT STEP "
                f"{step}/{MAX_AGENT_STEPS} ---"
            )

            response = await claude.messages.create(
                model=model,
                max_tokens=400,
                system=instructions,
                tools=claude_tools,
                messages=messages,
            )

            api_calls += 1
            total_input += response.usage.input_tokens
            total_output += response.usage.output_tokens

            print(
                "INPUT TOKENS:",
                response.usage.input_tokens,
            )
            print(
                "OUTPUT TOKENS:",
                response.usage.output_tokens,
            )
            print("STOP REASON:", response.stop_reason)

            messages.append(
                {
                    "role": "assistant",
                    "content": response.content,
                }
            )

            response_text = "\n".join(
                block.text
                for block in response.content
                if block.type == "text"
            )

            if response_text:
                print("\nMODEL TEXT:")
                print(response_text)

            if response.stop_reason == "end_turn":
                final_text = response_text.strip()

                missing_results = (
                    set(ALLOWED_TOOLS)
                    - successful_tools
                )

                if missing_results:
                    print(
                        "MISSING TOOL RESULTS:",
                        sorted(missing_results),
                    )
                    agent_status = (
                        "MISSING_TOOL_RESULTS"
                    )
                elif not final_text:
                    agent_status = (
                        "EMPTY_FINAL_RESPONSE"
                    )
                else:
                    agent_status = "MODEL_FINISHED"

                break

            if response.stop_reason != "tool_use":
                agent_status = (
                    f"STOPPED_{response.stop_reason}"
                ).upper()
                break

            requests = [
                block
                for block in response.content
                if block.type == "tool_use"
            ]

            if not requests:
                agent_status = (
                    "INVALID_TOOL_RESPONSE"
                )
                break

            if step == MAX_AGENT_STEPS:
                print(
                    "Step limit reached; tool requests "
                    "were not executed."
                )
                break

            tool_results = []

            for request in requests:
                tool_calls += 1

                print(
                    "\nMODEL REQUESTED:",
                    request.name,
                    json.dumps(request.input),
                )

                allowed = (
                    is_tool_allowed(
                        request.name,
                        ALLOWED_TOOLS,
                    )
                    and request.name in exposed_tools
                )

                if not allowed:
                    rejected_tool_calls += 1
                    is_error = True
                    result_data = None
                    result_text = json.dumps(
                        {
                            "error": (
                                "TOOL_NOT_ALLOWED"
                            ),
                            "tool": request.name,
                        }
                    )
                    print(
                        "APPLICATION TOOL CONTROL: "
                        "BLOCKED"
                    )
                else:
                    result = (
                        await mcp_client.call_tool(
                            request.name,
                            request.input,
                        )
                    )

                    is_error = bool(result.is_error)
                    result_text, result_data = (
                        mcp_result_payload(result)
                    )

                    print(
                        "APPLICATION TOOL CONTROL: "
                        "ALLOWED"
                    )

                print("MCP ERROR:", is_error)
                print("MCP RESULT:", result_text)

                if not is_error:
                    successful_tools.add(
                        request.name
                    )

                    if isinstance(result_data, dict):
                        tool_outputs[
                            request.name
                        ] = result_data

                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": request.id,
                        "content": result_text,
                        "is_error": is_error,
                    }
                )

            messages.append(
                {
                    "role": "user",
                    "content": tool_results,
                }
            )

        print("\nAGENT STATUS:", agent_status)
        print("AGENT STEPS:", api_calls)
        print("MCP TOOL REQUESTS:", tool_calls)
        print(
            "REJECTED TOOL REQUESTS:",
            rejected_tool_calls,
        )
        print("TOTAL INPUT TOKENS:", total_input)
        print("TOTAL OUTPUT TOKENS:", total_output)

        if agent_status != "MODEL_FINISHED":
            raise RuntimeError(
                "Controlled agent stopped before "
                "a valid final answer."
            )

        compliance = tool_outputs.get(
            "compliance_rules"
        )

        if not isinstance(compliance, dict):
            raise RuntimeError(
                "No parseable compliance result "
                "was retained."
            )

        model_action = extract_recommended_action(
            final_text
        )

        if not model_action:
            model_action = (
                "MISSING_RECOMMENDATION"
            )

        policy_next_step = str(
            compliance.get("next_step", "")
        ).strip().upper()

        model_matches_policy = (
            model_action == policy_next_step
        )

        authorization = (
            authorize_underwriting_step(
                case_id="CASE-001",
                requested_action=model_action,
                compliance=compliance,
                human_approval_supplied=False,
            )
        )

        consequential_test = (
            authorize_underwriting_step(
                case_id="CASE-001",
                requested_action="BIND_COVERAGE",
                compliance=compliance,
                human_approval_supplied=False,
            )
        )

        print(
            "\nMODEL RECOMMENDED ACTION:",
            model_action,
        )
        print(
            "POLICY NEXT STEP:",
            policy_next_step,
        )
        print(
            "MODEL MATCHES POLICY:",
            model_matches_policy,
        )

        print(
            "\nAPPLICATION AUTHORIZATION RESULT:"
        )
        print(
            json.dumps(
                authorization,
                indent=2,
            )
        )

        print(
            "\nHUMAN AUTHORIZATION BOUNDARY TEST:"
        )
        print(
            json.dumps(
                consequential_test,
                indent=2,
            )
        )

        checks_passed = (
            model_matches_policy
            and authorization[
                "authorization_status"
            ] == "PERMITTED"
            and authorization[
                "external_action_executed"
            ] is False
            and authorization[
                "coverage_decision_executed"
            ] is False
            and consequential_test[
                "authorization_status"
            ] == "BLOCKED"
            and consequential_test[
                "reason_code"
            ] == "HUMAN_APPROVAL_REQUIRED"
            and api_calls <= MAX_AGENT_STEPS
        )

        if not checks_passed:
            raise RuntimeError(
                "One or more controlled-agent "
                "checks failed."
            )

        print(
            "\nRUN STATUS: "
            "CONTROLLED_AGENT_FINISHED"
        )


if __name__ == "__main__":
    anyio.run(main)
