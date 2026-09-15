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
from v2.mcp_server.underwriting_server import mcp
from v2.services.telemetry import record_event


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PRICING = {
    "claude-haiku-4-5-20251001": {
        "input_per_million": 1.00,
        "output_per_million": 5.00,
    },
    "claude-haiku-4-5": {
        "input_per_million": 1.00,
        "output_per_million": 5.00,
    },
}

PRICE_SOURCE = (
    "https://docs.anthropic.com/"
    "en/docs/about-claude/pricing"
)

OUTPUT_CONTRACT = """
Return only these eight lines:

CASE_ID: CASE-001
EVIDENCE_STATUS: INCOMPLETE
OUTSTANDING_EVIDENCE: wind_mitigation, contractor_loss_history
POLICY_NEXT_STEP: REQUEST_EVIDENCE
HUMAN_REVIEW_REQUIRED: true
HUMAN_APPROVAL_REQUIRED: true
AUTHORIZATION_BOUNDARY: RECOMMENDATION_ONLY
RECOMMENDED_NEXT_STEP: REQUEST_EVIDENCE
""".strip()


def extract_mcp_payload(
    result: Any,
) -> dict[str, Any]:
    if result.is_error:
        raise RuntimeError(
            "MCP tool returned an error."
        )

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
        "No usable MCP dictionary payload."
    )


def call_model(
    client: Anthropic,
    model: str,
    label: str,
    system: str,
    prompt: str,
    max_tokens: int,
) -> dict[str, Any]:
    started = time.perf_counter()

    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    latency_ms = (
        time.perf_counter() - started
    ) * 1000

    text = "\n".join(
        block.text
        for block in response.content
        if block.type == "text"
    ).strip()

    result = {
        "label": label,
        "model_id": model,
        "text": text,
        "input_tokens": (
            response.usage.input_tokens
        ),
        "output_tokens": (
            response.usage.output_tokens
        ),
        "latency_ms": round(
            latency_ms,
            2,
        ),
        "stop_reason": str(
            response.stop_reason
        ),
    }

    print(f"\n--- {label} ---")
    print(
        "INPUT TOKENS:",
        result["input_tokens"],
    )
    print(
        "OUTPUT TOKENS:",
        result["output_tokens"],
    )
    print(
        "LATENCY MS:",
        result["latency_ms"],
    )
    print(
        "STOP REASON:",
        result["stop_reason"],
    )
    print("\nRESPONSE:")
    print(text)

    if result["stop_reason"] != "end_turn":
        raise RuntimeError(
            f"{label} did not finish normally."
        )

    return result


def read_field(
    text: str,
    field_name: str,
) -> str:
    target = field_name.upper() + ":"

    for raw_line in text.splitlines():
        cleaned = (
            raw_line.strip()
            .lstrip("#- ")
            .replace("*", "")
            .replace("`", "")
        )

        if cleaned.upper().startswith(target):
            return cleaned[
                len(target):
            ].strip()

    return ""


def evidence_names(value: str) -> set[str]:
    return set(
        re.findall(
            r"[a-z]+(?:_[a-z]+)+",
            value.lower(),
        )
    )


def validate_case_report(
    text: str,
) -> dict[str, Any]:
    checks = {
        "case_id": (
            read_field(text, "CASE_ID")
            == "CASE-001"
        ),
        "evidence_status": (
            read_field(
                text,
                "EVIDENCE_STATUS",
            ).upper()
            == "INCOMPLETE"
        ),
        "outstanding_evidence": (
            evidence_names(
                read_field(
                    text,
                    "OUTSTANDING_EVIDENCE",
                )
            )
            == {
                "wind_mitigation",
                "contractor_loss_history",
            }
        ),
    }

    return {
        "checks": checks,
        "valid": all(checks.values()),
    }


def validate_policy_report(
    text: str,
) -> dict[str, Any]:
    checks = {
        "policy_id": (
            read_field(text, "POLICY_ID")
            == "LAB-UW-POLICY-001"
        ),
        "policy_next_step": (
            read_field(
                text,
                "POLICY_NEXT_STEP",
            ).upper()
            == "REQUEST_EVIDENCE"
        ),
        "human_review": (
            read_field(
                text,
                "HUMAN_REVIEW_REQUIRED",
            ).lower()
            == "true"
        ),
        "human_approval": (
            read_field(
                text,
                "HUMAN_APPROVAL_REQUIRED",
            ).lower()
            == "true"
        ),
    }

    return {
        "checks": checks,
        "valid": all(checks.values()),
    }


def validate_final_output(
    text: str,
) -> dict[str, Any]:
    case_validation = validate_case_report(
        text
    )

    checks = {
        **case_validation["checks"],
        "policy_next_step": (
            read_field(
                text,
                "POLICY_NEXT_STEP",
            ).upper()
            == "REQUEST_EVIDENCE"
        ),
        "human_review": (
            read_field(
                text,
                "HUMAN_REVIEW_REQUIRED",
            ).lower()
            == "true"
        ),
        "human_approval": (
            read_field(
                text,
                "HUMAN_APPROVAL_REQUIRED",
            ).lower()
            == "true"
        ),
        "authorization_boundary": (
            read_field(
                text,
                "AUTHORIZATION_BOUNDARY",
            ).upper()
            == "RECOMMENDATION_ONLY"
        ),
        "recommendation": (
            read_field(
                text,
                "RECOMMENDED_NEXT_STEP",
            ).upper()
            == "REQUEST_EVIDENCE"
        ),
    }

    return {
        "checks": checks,
        "valid": all(checks.values()),
    }


def calculate_economics(
    runs: list[dict[str, Any]],
    rates: dict[str, float],
) -> dict[str, Any]:
    input_tokens = sum(
        run["input_tokens"]
        for run in runs
    )
    output_tokens = sum(
        run["output_tokens"]
        for run in runs
    )
    latency_ms = sum(
        run["latency_ms"]
        for run in runs
    )

    input_cost = (
        input_tokens
        / 1_000_000
        * rates["input_per_million"]
    )
    output_cost = (
        output_tokens
        / 1_000_000
        * rates["output_per_million"]
    )

    return {
        "model_calls": len(runs),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": (
            input_tokens + output_tokens
        ),
        "sequential_latency_ms": round(
            latency_ms,
            2,
        ),
        "estimated_input_cost_usd": round(
            input_cost,
            8,
        ),
        "estimated_output_cost_usd": round(
            output_cost,
            8,
        ),
        "estimated_total_cost_usd": round(
            input_cost + output_cost,
            8,
        ),
    }


def latest_lab8_event() -> dict[str, Any]:
    telemetry_file = (
        PROJECT_ROOT
        / "v2"
        / "data"
        / "interaction_telemetry.jsonl"
    )

    events = [
        json.loads(line)
        for line in telemetry_file.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    matching = [
        event
        for event in events
        if event.get("lab")
        == "V2_LAB_08"
    ]

    if not matching:
        raise RuntimeError(
            "Lab 8 telemetry was not found."
        )

    return matching[-1]


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

    if not api_key or not model:
        raise SystemExit(
            "Claude configuration is missing."
        )

    rates = MODEL_PRICING.get(model)

    if rates is None:
        raise SystemExit(
            "No verified pricing configured "
            f"for model: {model}"
        )

    claude = Anthropic(api_key=api_key)

    mcp_started = time.perf_counter()

    async with Client(
        mcp,
        raise_exceptions=True,
    ) as mcp_client:
        listed = await mcp_client.list_tools()
        discovered = {
            tool.name
            for tool in listed.tools
        }

        if discovered != set(
            DEFAULT_ALLOWED_TOOLS
        ):
            raise RuntimeError(
                "MCP tool contract changed."
            )

        submission_result = (
            await mcp_client.call_tool(
                "submission",
                {"case_id": "CASE-001"},
            )
        )
        compliance_result = (
            await mcp_client.call_tool(
                "compliance_rules",
                {"case_id": "CASE-001"},
            )
        )

    mcp_latency_ms = (
        time.perf_counter() - mcp_started
    ) * 1000

    submission = extract_mcp_payload(
        submission_result
    )
    compliance = extract_mcp_payload(
        compliance_result
    )

    authoritative_bundle = {
        "submission": submission,
        "compliance_rules": compliance,
    }

    bundle_json = json.dumps(
        authoritative_bundle,
        indent=2,
        sort_keys=True,
    )

    single_system = """
You are one controlled underwriting analyst.
Use only the supplied authoritative JSON.
Treat policy values and evidence names as opaque.
Do not add external insurance knowledge.
Do not approve, decline, price, or bind coverage.
Do not call the policy next step authorized.
Return only the required output contract.
""".strip()

    single_prompt = (
        OUTPUT_CONTRACT
        + "\n\nAUTHORITATIVE JSON:\n"
        + bundle_json
    )

    single_run = call_model(
        claude,
        model,
        "SINGLE AGENT",
        single_system,
        single_prompt,
        350,
    )

    case_system = """
You are a case-state extraction specialist.
Use only the supplied submission JSON.
Do not interpret risk or make recommendations.
Return only the requested three lines.
""".strip()

    case_prompt = (
        "Return only:\n"
        "CASE_ID: <value>\n"
        "EVIDENCE_STATUS: <value>\n"
        "OUTSTANDING_EVIDENCE: "
        "<comma-separated exact names>\n\n"
        "SUBMISSION JSON:\n"
        + json.dumps(
            submission,
            indent=2,
            sort_keys=True,
        )
    )

    case_run = call_model(
        claude,
        model,
        "CASE SPECIALIST",
        case_system,
        case_prompt,
        200,
    )

    policy_system = """
You are a policy-state extraction specialist.
Use only the supplied compliance JSON.
Do not expand, interpret, or authorize the policy.
Return only the requested four lines.
""".strip()

    policy_prompt = (
        "Return only:\n"
        "POLICY_ID: <value>\n"
        "POLICY_NEXT_STEP: <value>\n"
        "HUMAN_REVIEW_REQUIRED: <true-or-false>\n"
        "HUMAN_APPROVAL_REQUIRED: <true-or-false>\n\n"
        "COMPLIANCE JSON:\n"
        + json.dumps(
            compliance,
            indent=2,
            sort_keys=True,
        )
    )

    policy_run = call_model(
        claude,
        model,
        "POLICY SPECIALIST",
        policy_system,
        policy_prompt,
        200,
    )

    coordinator_system = """
You are a controlled workflow coordinator.
Use only the authoritative JSON and specialist
reports supplied below.

Specialist reports are advisory extracts.
The authoritative JSON wins if they conflict.

Do not add external insurance knowledge.
Do not approve, decline, price, or bind coverage.
Call the policy result policy-directed, not
authorized. Return only the output contract.
""".strip()

    coordinator_prompt = (
        OUTPUT_CONTRACT
        + "\n\nAUTHORITATIVE JSON:\n"
        + bundle_json
        + "\n\nCASE SPECIALIST REPORT:\n"
        + case_run["text"]
        + "\n\nPOLICY SPECIALIST REPORT:\n"
        + policy_run["text"]
    )

    coordinator_run = call_model(
        claude,
        model,
        "MULTI-AGENT COORDINATOR",
        coordinator_system,
        coordinator_prompt,
        350,
    )

    single_validation = (
        validate_final_output(
            single_run["text"]
        )
    )
    case_validation = (
        validate_case_report(
            case_run["text"]
        )
    )
    policy_validation = (
        validate_policy_report(
            policy_run["text"]
        )
    )
    coordinator_validation = (
        validate_final_output(
            coordinator_run["text"]
        )
    )

    multi_team_valid = all(
        (
            case_validation["valid"],
            policy_validation["valid"],
            coordinator_validation["valid"],
        )
    )

    single_economics = (
        calculate_economics(
            [single_run],
            rates,
        )
    )
    multi_economics = (
        calculate_economics(
            [
                case_run,
                policy_run,
                coordinator_run,
            ],
            rates,
        )
    )

    single_action = read_field(
        single_run["text"],
        "RECOMMENDED_NEXT_STEP",
    ).upper()
    multi_action = read_field(
        coordinator_run["text"],
        "RECOMMENDED_NEXT_STEP",
    ).upper()

    single_authorization = (
        authorize_underwriting_step(
            case_id="CASE-001",
            requested_action=single_action,
            compliance=compliance,
            human_approval_supplied=False,
        )
    )
    multi_authorization = (
        authorize_underwriting_step(
            case_id="CASE-001",
            requested_action=multi_action,
            compliance=compliance,
            human_approval_supplied=False,
        )
    )

    cost_ratio = (
        multi_economics[
            "estimated_total_cost_usd"
        ]
        / single_economics[
            "estimated_total_cost_usd"
        ]
    )

    latency_ratio = (
        multi_economics[
            "sequential_latency_ms"
        ]
        / single_economics[
            "sequential_latency_ms"
        ]
    )

    both_outcomes_valid = (
        single_validation["valid"]
        and multi_team_valid
    )

    if (
        both_outcomes_valid
        and multi_economics["model_calls"]
        > single_economics["model_calls"]
        and multi_economics[
            "estimated_total_cost_usd"
        ]
        > single_economics[
            "estimated_total_cost_usd"
        ]
    ):
        architecture_finding = (
            "SINGLE_AGENT_PREFERRED_FOR_CASE_001"
        )
    elif both_outcomes_valid:
        architecture_finding = (
            "NO_CLEAR_ECONOMIC_WINNER"
        )
    else:
        architecture_finding = (
            "OUTCOME_VALIDATION_FAILED"
        )

    lab8 = latest_lab8_event()
    lab8_cost = (
        lab8["input_tokens"]
        / 1_000_000
        * rates["input_per_million"]
        + lab8["output_tokens"]
        / 1_000_000
        * rates["output_per_million"]
    )

    no_actions = all(
        (
            single_authorization.get(
                "external_action_executed"
            )
            is False,
            single_authorization.get(
                "coverage_decision_executed"
            )
            is False,
            multi_authorization.get(
                "external_action_executed"
            )
            is False,
            multi_authorization.get(
                "coverage_decision_executed"
            )
            is False,
        )
    )

    lab_verified = all(
        (
            both_outcomes_valid,
            single_action
            == multi_action
            == compliance["next_step"]
            == "REQUEST_EVIDENCE",
            no_actions,
            discovered
            == set(DEFAULT_ALLOWED_TOOLS),
        )
    )

    report = {
        "lab": "V2_LAB_09",
        "experiment_scope": (
            "ONE_CASE_ONE_RUN_NOT_STATISTICAL"
        ),
        "model_id": model,
        "price_source": PRICE_SOURCE,
        "pricing": rates,
        "shared_mcp_calls": 2,
        "shared_mcp_latency_ms": round(
            mcp_latency_ms,
            2,
        ),
        "lab8_controlled_agent_baseline": {
            "model_calls": (
                lab8["model_calls"]
            ),
            "input_tokens": (
                lab8["input_tokens"]
            ),
            "output_tokens": (
                lab8["output_tokens"]
            ),
            "estimated_model_cost_usd": round(
                lab8_cost,
                8,
            ),
        },
        "single_agent": {
            "validation": single_validation,
            "economics": single_economics,
            "authorization_status": (
                single_authorization.get(
                    "authorization_status"
                )
            ),
        },
        "multi_agent": {
            "case_specialist_validation": (
                case_validation
            ),
            "policy_specialist_validation": (
                policy_validation
            ),
            "coordinator_validation": (
                coordinator_validation
            ),
            "team_valid": multi_team_valid,
            "economics": multi_economics,
            "authorization_status": (
                multi_authorization.get(
                    "authorization_status"
                )
            ),
        },
        "comparison": {
            "same_physical_model": True,
            "same_authoritative_data": True,
            "same_output_contract": True,
            "both_outcomes_valid": (
                both_outcomes_valid
            ),
            "multi_to_single_cost_ratio": round(
                cost_ratio,
                3,
            ),
            "multi_to_single_sequential_latency_ratio": (
                round(latency_ratio, 3)
            ),
            "architecture_finding": (
                architecture_finding
            ),
            "external_actions_executed": 0,
        },
        "lab_verified": lab_verified,
    }

    results_file = (
        PROJECT_ROOT
        / "v2"
        / "data"
        / "lab09_economics_results.json"
    )
    results_file.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    record_event(
        {
            "lab": "V2_LAB_09",
            "route": (
                "MULTI_AGENT_ECONOMICS"
            ),
            "model_id": model,
            "input_tokens": (
                single_economics[
                    "input_tokens"
                ]
                + multi_economics[
                    "input_tokens"
                ]
            ),
            "output_tokens": (
                single_economics[
                    "output_tokens"
                ]
                + multi_economics[
                    "output_tokens"
                ]
            ),
            "model_calls": 4,
            "tool_calls": 2,
            "mcp_tool_calls": 2,
            "stop_reason": "end_turn",
            "single_agent_valid": (
                single_validation["valid"]
            ),
            "multi_agent_valid": (
                multi_team_valid
            ),
            "architecture_finding": (
                architecture_finding
            ),
            "external_action_executed": False,
            "coverage_decision_executed": False,
            "lab_verified": lab_verified,
        }
    )

    print("\nSINGLE AGENT VALIDATION:")
    print(
        json.dumps(
            single_validation,
            indent=2,
        )
    )

    print("\nMULTI-AGENT VALIDATION:")
    print(
        json.dumps(
            {
                "case_specialist": (
                    case_validation
                ),
                "policy_specialist": (
                    policy_validation
                ),
                "coordinator": (
                    coordinator_validation
                ),
                "team_valid": multi_team_valid,
            },
            indent=2,
        )
    )

    print("\nECONOMIC COMPARISON:")
    print(
        json.dumps(
            report["comparison"],
            indent=2,
        )
    )

    print("\nSINGLE AGENT ECONOMICS:")
    print(
        json.dumps(
            single_economics,
            indent=2,
        )
    )

    print("\nMULTI-AGENT ECONOMICS:")
    print(
        json.dumps(
            multi_economics,
            indent=2,
        )
    )

    print(
        "\nRESULT FILE:",
        results_file.name,
    )
    print("SECRETS PRINTED: False")

    if not lab_verified:
        raise RuntimeError(
            "Lab 9 validation failed. "
            "Paste the complete output."
        )

    print("V2 LAB 9 STATUS: COMPLETE")


if __name__ == "__main__":
    anyio.run(main)
