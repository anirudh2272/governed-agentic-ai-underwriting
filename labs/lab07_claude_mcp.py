import json
import os
import sys
from pathlib import Path

import anyio
from anthropic import AsyncAnthropic
from dotenv import load_dotenv
from mcp import Client, StdioServerParameters
from mcp.types import TextContent

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

claude = AsyncAnthropic(timeout=30.0, max_retries=0)
model = os.environ["CLAUDE_MODEL"].strip()

MAX_AGENT_STEPS = 6

ALLOWED_TOOLS = {
    "submission",
    "weather_risk",
    "compliance_rules",
    "loss_history",
    "base_score",
}


def mcp_result_text(result):
    text_parts = [
        block.text
        for block in result.content
        if isinstance(block, TextContent)
    ]

    if text_parts:
        return "\n".join(text_parts)

    if result.structured_content is not None:
        return json.dumps(result.structured_content)

    return "MCP tool returned no readable content."


async def main():
    server = StdioServerParameters(
        command=sys.executable,
        args=["-m", "mcp_server.underwriting_server"],
        env={"PYTHONPATH": str(PROJECT_ROOT)},
    )

    async with Client(server) as mcp_client:
        listed = await mcp_client.list_tools()

        discovered = {
            tool.name: tool
            for tool in listed.tools
            if tool.name in ALLOWED_TOOLS
        }

        missing = ALLOWED_TOOLS - set(discovered)

        if missing:
            raise RuntimeError(
                f"Required MCP tools missing: {sorted(missing)}"
            )

        claude_tools = [
            {
                "name": tool.name,
                "description": tool.description or tool.name,
                "input_schema": tool.input_schema,
            }
            for tool in discovered.values()
        ]

        print("MCP CONNECTION: SUCCESS")
        print("TOOLS PROVIDED TO CLAUDE:", sorted(discovered))

        messages = [
            {
                "role": "user",
                "content": (
                    "Investigate synthetic underwriting case CASE-001 "
                    "and recommend the policy-controlled next step."
                ),
            }
        ]

        instructions = (
            "Use submission first. Then call weather_risk, "
            "compliance_rules, loss_history, and base_score using "
            "identifiers returned by tools. Retrieve all five results. "
            "Follow compliance next_step exactly. Missing loss-history "
            "evidence does not mean zero losses. The score is based on "
            "invented training weights. Name only evidence requirements "
            "returned by tools; do not invent document periods or forms. "
            "Give a recommendation for a human underwriter. Do not bind "
            "coverage, set premium, approve, or decline the case. "
            "Keep the final response under 200 words."
        )

        successful_tools = set()
        api_calls = 0
        tool_calls = 0
        total_input = 0
        total_output = 0
        run_status = "STEP_LIMIT_REACHED"

        for step in range(1, MAX_AGENT_STEPS + 1):
            print(
                f"\n--- CLAUDE/MCP STEP "
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

            print("INPUT TOKENS:", response.usage.input_tokens)
            print("OUTPUT TOKENS:", response.usage.output_tokens)
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
                missing_results = (
                    ALLOWED_TOOLS - successful_tools
                )

                if missing_results:
                    print(
                        "MISSING TOOL RESULTS:",
                        sorted(missing_results),
                    )
                    run_status = "MISSING_TOOL_RESULTS"
                elif response_text.strip():
                    run_status = "MODEL_FINISHED"
                else:
                    run_status = "EMPTY_FINAL_RESPONSE"

                break

            if response.stop_reason != "tool_use":
                run_status = (
                    f"STOPPED_{response.stop_reason}".upper()
                )
                break

            requests = [
                block
                for block in response.content
                if block.type == "tool_use"
            ]

            if not requests:
                run_status = "INVALID_TOOL_RESPONSE"
                break

            if step == MAX_AGENT_STEPS:
                print(
                    "No request budget remains; "
                    "tools were not executed."
                )
                break

            tool_results = []

            for request in requests:
                tool_calls += 1

                print(
                    "\nCLAUDE REQUESTED:",
                    request.name,
                    json.dumps(request.input),
                )

                if request.name not in discovered:
                    result_text = "Tool not allowed"
                    is_error = True
                else:
                    result = await mcp_client.call_tool(
                        request.name,
                        request.input,
                    )
                    result_text = mcp_result_text(result)
                    is_error = bool(result.is_error)

                print("MCP ERROR:", is_error)
                print("MCP RESULT:", result_text)

                if not is_error:
                    successful_tools.add(request.name)

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

        print("\nRUN STATUS:", run_status)
        print("CLAUDE API CALLS:", api_calls)
        print("MCP TOOL CALLS:", tool_calls)
        print("TOTAL INPUT TOKENS:", total_input)
        print("TOTAL OUTPUT TOKENS:", total_output)

        if run_status != "MODEL_FINISHED":
            raise RuntimeError(
                "Lab 7 needs review. Paste all output."
            )


if __name__ == "__main__":
    anyio.run(main)
