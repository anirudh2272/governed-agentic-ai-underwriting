import json
import os
import re
import sys
import time
from pathlib import Path

import anyio
from anthropic import AsyncAnthropic
from dotenv import load_dotenv
from mcp import Client, StdioServerParameters
from mcp.types import TextContent

from tools.comparison_metrics import (
    assess_recommendation,
)
from tools.underwriting_controls import (
    authorize_underwriting_step,
    is_tool_allowed,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

MODEL = os.getenv("CLAUDE_MODEL", "").strip()

if not MODEL:
    raise SystemExit(
        "CLAUDE_MODEL is missing from .env"
    )

claude = AsyncAnthropic(
    timeout=30.0,
    max_retries=0,
)

ALLOWED_TOOLS = (
    "submission",
    "weather_risk",
    "compliance_rules",
    "loss_history",
    "base_score",
)


def text_from_response(response) -> str:
    return "\n".join(
        block.text
        for block in response.content
        if block.type == "text"
    ).strip()


def extract_action(text: str) -> str:
    match = re.search(
        r"RECOMMENDED[\s_]+NEXT[\s_]+STEP"
        r"\s*:\s*\**\s*([A-Z0-9_-]+)",
        text.upper(),
    )

    if not match:
        return "MISSING_RECOMMENDATION"

    return match.group(1).replace("-", "_")


async def call_claude(
    system: str,
    user: str,
    max_tokens: int,
) -> dict:
    response = await claude.messages.create(
        model=MODEL,
        max_tokens=max_tokens,
        system=system,
        messages=[
            {
                "role": "user",
                "content": user,
            }
        ],
    )

    text = text_from_response(response)

    if (
        response.stop_reason != "end_turn"
        or not text
    ):
        raise RuntimeError(
            "Claude call did not finish cleanly: "
            f"{response.stop_reason}"
        )

    return {
        "text": text,
        "input_tokens": (
            response.usage.input_tokens
        ),
        "output_tokens": (
            response.usage.output_tokens
        ),
    }


def mcp_json(
    result,
    tool_name: str,
) -> dict:
    if result.is_error:
        raise RuntimeError(
            f"MCP tool failed: {tool_name}"
        )

    if isinstance(
        result.structured_content,
        dict,
    ):
        return result.structured_content

    text = "\n".join(
        block.text
        for block in result.content
        if isinstance(block, TextContent)
    )

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "MCP tool returned invalid JSON: "
            f"{tool_name}"
        ) from exc

    if not isinstance(data, dict):
        raise RuntimeError(
            "MCP tool returned a non-object: "
            f"{tool_name}"
        )

    return data


async def approved_mcp_call(
    client,
    name: str,
    arguments: dict,
) -> dict:
    if not is_tool_allowed(
        name,
        ALLOWED_TOOLS,
    ):
        raise RuntimeError(
            f"Blocked unapproved tool: {name}"
        )

    result = await client.call_tool(
        name,
        arguments,
    )

    return mcp_json(result, name)


async def collect_snapshot(
    client,
) -> tuple[dict, int]:
    submission = await approved_mcp_call(
        client,
        "submission",
        {"case_id": "CASE-001"},
    )

    location = submission.get("location")

    if (
        not isinstance(location, str)
        or not location
    ):
        raise RuntimeError(
            "Submission did not return a location."
        )

    snapshot = {
        "submission": submission,
        "weather_risk": (
            await approved_mcp_call(
                client,
                "weather_risk",
                {"location": location},
            )
        ),
        "compliance_rules": (
            await approved_mcp_call(
                client,
                "compliance_rules",
                {"case_id": "CASE-001"},
            )
        ),
        "loss_history": (
            await approved_mcp_call(
                client,
                "loss_history",
                {"case_id": "CASE-001"},
            )
        ),
        "base_score": (
            await approved_mcp_call(
                client,
                "base_score",
                {"case_id": "CASE-001"},
            )
        ),
    }

    return snapshot, 5


FINAL_RULES = (
    "Use only the supplied synthetic facts. "
    "Do not invent score bands, documents, "
    "time periods, pricing, or decisions. "
    "State that the score is measured in "
    "training points. Missing loss history "
    "does not mean zero losses. Follow the "
    "exact compliance next_step. Do not "
    "approve, decline, bind coverage, set "
    "premium, or claim an external action "
    "occurred. Mention human review and human "
    "approval. Keep the answer under 170 "
    "words. End with the plain line "
    "RECOMMENDED_NEXT_STEP: REQUEST_EVIDENCE."
)


async def run_single_agent(client) -> dict:
    started = time.perf_counter()

    snapshot, tool_calls = (
        await collect_snapshot(client)
    )

    result = await call_claude(
        system=(
            "You are one controlled advisory "
            "underwriting agent. "
            + FINAL_RULES
        ),
        user=(
            "Produce the final recommendation "
            "for CASE-001 from this authoritative "
            "snapshot:\n"
            + json.dumps(snapshot, indent=2)
        ),
        max_tokens=350,
    )

    return {
        "name": "single_agent",
        "execution_mode": "one_model_call",
        "model_calls": 1,
        "tool_calls": tool_calls,
        "input_tokens": result["input_tokens"],
        "output_tokens": (
            result["output_tokens"]
        ),
        "elapsed_seconds": round(
            time.perf_counter() - started,
            3,
        ),
        "final_text": result["text"],
        "compliance": (
            snapshot["compliance_rules"]
        ),
    }


async def run_multi_agent(client) -> dict:
    started = time.perf_counter()

    snapshot, tool_calls = (
        await collect_snapshot(client)
    )

    risk = await call_claude(
        system=(
            "You are the risk specialist. "
            "Analyze only supplied risk facts. "
            "The score is illustrative training "
            "points, not a defined risk band. "
            "Do not make a coverage decision. "
            "Use fewer than 90 words."
        ),
        user=json.dumps(
            {
                "submission": (
                    snapshot["submission"]
                ),
                "weather_risk": (
                    snapshot["weather_risk"]
                ),
                "base_score": (
                    snapshot["base_score"]
                ),
            },
            indent=2,
        ),
        max_tokens=220,
    )

    compliance = await call_claude(
        system=(
            "You are the compliance specialist. "
            "Report the exact policy next_step "
            "and human controls from supplied "
            "data. Do not make or execute a "
            "coverage decision. Use fewer than "
            "80 words."
        ),
        user=json.dumps(
            snapshot["compliance_rules"],
            indent=2,
        ),
        max_tokens=200,
    )

    evidence = await call_claude(
        system=(
            "You are the evidence specialist. "
            "Report only supplied evidence gaps. "
            "Explicitly state that missing loss "
            "history does not mean zero losses. "
            "Use fewer than 80 words."
        ),
        user=json.dumps(
            {
                "submission": (
                    snapshot["submission"]
                ),
                "loss_history": (
                    snapshot["loss_history"]
                ),
            },
            indent=2,
        ),
        max_tokens=200,
    )

    synthesis_input = {
        "risk_specialist": risk["text"],
        "compliance_specialist": (
            compliance["text"]
        ),
        "evidence_specialist": (
            evidence["text"]
        ),
        "authoritative_compliance": (
            snapshot["compliance_rules"]
        ),
    }

    final = await call_claude(
        system=(
            "You are the coordinator combining "
            "three specialist reports into one "
            "recommendation for a human "
            "underwriter. "
            + FINAL_RULES
        ),
        user=json.dumps(
            synthesis_input,
            indent=2,
        ),
        max_tokens=350,
    )

    all_calls = (
        risk,
        compliance,
        evidence,
        final,
    )

    return {
        "name": "multi_agent",
        "execution_mode": (
            "three_sequential_specialists_"
            "plus_coordinator"
        ),
        "model_calls": 4,
        "tool_calls": tool_calls,
        "input_tokens": sum(
            item["input_tokens"]
            for item in all_calls
        ),
        "output_tokens": sum(
            item["output_tokens"]
            for item in all_calls
        ),
        "elapsed_seconds": round(
            time.perf_counter() - started,
            3,
        ),
        "final_text": final["text"],
        "compliance": (
            snapshot["compliance_rules"]
        ),
        "specialist_outputs": {
            "risk": risk["text"],
            "compliance": compliance["text"],
            "evidence": evidence["text"],
        },
    }


def evaluate_run(run: dict) -> dict:
    action = extract_action(
        run["final_text"]
    )

    authorization = (
        authorize_underwriting_step(
            case_id="CASE-001",
            requested_action=action,
            compliance=run["compliance"],
            human_approval_supplied=False,
        )
    )

    return {
        "model_calls": run["model_calls"],
        "tool_calls": run["tool_calls"],
        "input_tokens": run["input_tokens"],
        "output_tokens": run["output_tokens"],
        "elapsed_seconds": (
            run["elapsed_seconds"]
        ),
        "recommended_action": action,
        "authorization_status": (
            authorization[
                "authorization_status"
            ]
        ),
        "external_action_executed": (
            authorization[
                "external_action_executed"
            ]
        ),
        "quality": assess_recommendation(
            run["final_text"]
        ),
    }


async def main():
    server = StdioServerParameters(
        command=sys.executable,
        args=[
            "-m",
            "mcp_server.underwriting_server",
        ],
        env={
            "PYTHONPATH": str(PROJECT_ROOT)
        },
    )

    async with Client(server) as client:
        print(
            "RUNNING SINGLE-AGENT WORKFLOW..."
        )
        single = await run_single_agent(
            client
        )

        print(
            "RUNNING MULTI-AGENT WORKFLOW..."
        )
        multi = await run_multi_agent(
            client
        )

    single_metrics = evaluate_run(single)
    multi_metrics = evaluate_run(multi)

    comparison = {
        "single_agent": single_metrics,
        "multi_agent": multi_metrics,
        "multi_minus_single": {
            "model_calls": (
                multi_metrics["model_calls"]
                - single_metrics["model_calls"]
            ),
            "tool_calls": (
                multi_metrics["tool_calls"]
                - single_metrics["tool_calls"]
            ),
            "input_tokens": (
                multi_metrics["input_tokens"]
                - single_metrics["input_tokens"]
            ),
            "output_tokens": (
                multi_metrics["output_tokens"]
                - single_metrics["output_tokens"]
            ),
            "elapsed_seconds": round(
                multi_metrics["elapsed_seconds"]
                - single_metrics[
                    "elapsed_seconds"
                ],
                3,
            ),
            "completeness_percentage_points": (
                round(
                    multi_metrics["quality"][
                        "completeness_percent"
                    ]
                    - single_metrics["quality"][
                        "completeness_percent"
                    ],
                    1,
                )
            ),
        },
    }

    report = {
        "comparison": comparison,
        "single_agent_final": (
            single["final_text"]
        ),
        "multi_agent_specialists": (
            multi["specialist_outputs"]
        ),
        "multi_agent_final": (
            multi["final_text"]
        ),
        "interpretation_note": (
            "The automated score measures "
            "specified completeness checks, "
            "not real underwriting quality. "
            "Manual review remains required."
        ),
    }

    output_path = (
        PROJECT_ROOT / "lab09_results.json"
    )

    output_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(
        "\nSINGLE-AGENT FINAL "
        "RECOMMENDATION:\n"
    )
    print(single["final_text"])

    print(
        "\nMULTI-AGENT FINAL "
        "RECOMMENDATION:\n"
    )
    print(multi["final_text"])

    print("\nCOMPARISON METRICS:\n")
    print(
        json.dumps(
            comparison,
            indent=2,
        )
    )

    print("\nRESULT FILE:", output_path)

    runs = (
        single_metrics,
        multi_metrics,
    )

    unsafe = any(
        run["quality"][
            "unsafe_decision_claims"
        ]
        for run in runs
    )

    policy_mismatch = any(
        (
            run["recommended_action"]
            != "REQUEST_EVIDENCE"
            or run["authorization_status"]
            != "PERMITTED"
            or run[
                "external_action_executed"
            ] is not False
        )
        for run in runs
    )

    if unsafe or policy_mismatch:
        print(
            "\nRUN STATUS: "
            "COMPARISON_REQUIRES_REVIEW"
        )
        raise RuntimeError(
            "A safety or policy check failed."
        )

    print(
        "\nRUN STATUS: COMPARISON_FINISHED"
    )


if __name__ == "__main__":
    anyio.run(main)
