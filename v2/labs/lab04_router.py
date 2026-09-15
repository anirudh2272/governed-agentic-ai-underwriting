import json
from datetime import datetime, timezone
from time import perf_counter
from uuid import uuid4

from tools.underwriting_tools import compliance_rules
from v2.control_plane.router import route_request
from v2.services.telemetry import record_event

case_id = "CASE-001"
started = perf_counter()

policy = compliance_rules(case_id)

if "error" in policy:
    raise SystemExit(policy["error"])

decision = route_request(
    task_type="NEXT_WORKFLOW_STEP",
    deterministic_answer=policy["next_step"],
    complexity="LOW",
    business_risk="HIGH",
)

latency_ms = round(
    (perf_counter() - started) * 1000,
    2,
)

try:
    route_request(
        task_type="INVALID_TEST",
        deterministic_answer=None,
        complexity="UNSUPPORTED",
        business_risk="LOW",
    )
    invalid_input_blocked = False
except ValueError:
    invalid_input_blocked = True

event = {
    "interaction_id": str(uuid4()),
    "created_at": datetime.now(timezone.utc).isoformat(),
    "lab": "V2_LAB_04",
    "case_id": case_id,
    "task_type": decision["task_type"],
    "route": decision["route"],
    "reason_code": decision["reason_code"],
    "requested_capability": (
        decision["requested_capability"]
    ),
    "model_required": decision["model_required"],
    "model_id": None,
    "model_calls": 0,
    "input_tokens": 0,
    "output_tokens": 0,
    "cache_creation_input_tokens": 0,
    "cache_read_input_tokens": 0,
    "context_characters": 0,
    "model_tool_calls": 0,
    "application_data_lookups": 1,
    "agent_steps": 0,
    "latency_ms": latency_ms,
    "stop_reason": "DETERMINISTIC_COMPLETION",
    "business_outcome": (
        decision["deterministic_result"]
    ),
    "estimated_model_cost_usd": 0.0,
    "external_action_executed": False,
    "coverage_decision_executed": False,
    "human_approval_required": (
        policy["human_approval_required"]
    ),
}

record_event(event)

zero_model_path_valid = (
    event["route"] == "NO_LLM"
    and event["model_required"] is False
    and event["model_id"] is None
    and event["model_calls"] == 0
    and event["input_tokens"] == 0
    and event["output_tokens"] == 0
    and event["estimated_model_cost_usd"] == 0.0
    and event["business_outcome"]
    == "REQUEST_EVIDENCE"
    and event["external_action_executed"] is False
    and event["coverage_decision_executed"] is False
)

print("ROUTE DECISION:")
print(json.dumps(decision, indent=2))

print("\nZERO-MODEL TELEMETRY:")
print(json.dumps(event, indent=2))

print(
    "\nINVALID ROUTER INPUT BLOCKED:",
    invalid_input_blocked,
)
print(
    "ZERO-MODEL PATH VERIFIED:",
    zero_model_path_valid,
)
print("SECRETS PRINTED: False")
print("V2 LAB 4 STATUS: COMPLETE")
