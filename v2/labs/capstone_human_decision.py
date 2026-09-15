import hashlib
import json
import os
from pathlib import Path

from tools.underwriting_tools import (
    compliance_rules,
    submission,
)
from v2.services import case_state
from v2.services.case_state import (
    record_human_decision,
)
from v2.services.telemetry import (
    TELEMETRY_FILE,
    record_event,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "underwriting_cases.json"
)
BACKUP_FILE = (
    PROJECT_ROOT
    / "v2"
    / "data"
    / "capstone_pre_human_decision_backup.json"
)
PHASE_B_FILE = (
    PROJECT_ROOT
    / "v2"
    / "data"
    / "capstone_phase_b_recommendation.json"
)
RESULT_FILE = (
    PROJECT_ROOT
    / "v2"
    / "data"
    / "capstone_human_decision.json"
)

CASE_ID = "CASE-001"
DECISION_ID = "CAPSTONE-CASE001-DECISION-001"
HUMAN_DECISION = "REFER"
AI_RECOMMENDATION = "REFER"
HUMAN_REASON = (
    "High coastal and hurricane exposure, "
    "flood zone AE, partial wind mitigation, "
    "and two verified losses require "
    "specialist underwriting judgment."
)
HUMAN_ACTOR = "LAB_HUMAN_REVIEWER"
DECISION_SOURCE = "EXPLICIT_USER_INPUT"
TELEMETRY_EVENT_ID = (
    "CAPSTONE-CASE001-HUMAN-DECISION-001"
)


def load_json(path: Path) -> dict:
    value = json.loads(
        path.read_text(encoding="utf-8")
    )

    if not isinstance(value, dict):
        raise RuntimeError(
            f"{path.name} must contain a JSON object"
        )

    return value


def file_hash(path: Path) -> str:
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def write_json_atomic(
    path: Path,
    payload: dict,
) -> None:
    temporary = path.with_name(
        path.name + ".tmp"
    )
    temporary.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def telemetry_event_exists(
    event_id: str,
) -> bool:
    if not TELEMETRY_FILE.exists():
        return False

    for line in TELEMETRY_FILE.read_text(
        encoding="utf-8"
    ).splitlines():
        if not line.strip():
            continue

        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue

        if (
            isinstance(event, dict)
            and event.get("event_id") == event_id
        ):
            return True

    return False


for required_file in (
    DATA_FILE,
    BACKUP_FILE,
    PHASE_B_FILE,
):
    if not required_file.is_file():
        raise RuntimeError(
            f"Required file is missing: {required_file}"
        )

phase_b = load_json(PHASE_B_FILE)
backup_cases = load_json(BACKUP_FILE)
before_cases = load_json(DATA_FILE)

backup_case = backup_cases.get(CASE_ID)
before_case = before_cases.get(CASE_ID)

if not isinstance(backup_case, dict):
    raise RuntimeError(
        "CASE-001 is absent from the rollback file"
    )

if not isinstance(before_case, dict):
    raise RuntimeError(
        "CASE-001 is absent from authoritative state"
    )

phase_b_valid = all(
    (
        phase_b.get("recommendation")
        == AI_RECOMMENDATION,
        isinstance(
            phase_b.get("validation"),
            dict,
        ),
        phase_b.get(
            "validation",
            {},
        ).get("phase_b_verified") is True,
        phase_b.get("authorization_status")
        == "PENDING_HUMAN_DECISION",
        phase_b.get(
            "external_action_executed"
        )
        is False,
    )
)

if not phase_b_valid:
    raise RuntimeError(
        "The saved Phase B recommendation is invalid"
    )

backup_hash_before = file_hash(BACKUP_FILE)
before_hash = file_hash(DATA_FILE)
backup_hash = file_hash(BACKUP_FILE)

existing_decision = before_case.get(
    "human_decision"
)
fresh_decision = not isinstance(
    existing_decision,
    dict,
)

if fresh_decision and before_hash != backup_hash:
    raise RuntimeError(
        "Current authoritative state does not match "
        "the pre-human-decision backup"
    )

decision_arguments = {
    "decision_id": DECISION_ID,
    "decision": HUMAN_DECISION,
    "reason": HUMAN_REASON,
    "human_actor": HUMAN_ACTOR,
    "human_confirmation": True,
    "source": DECISION_SOURCE,
    "ai_recommendation": AI_RECOMMENDATION,
}

try:
    write_result = record_human_decision(
        CASE_ID,
        **decision_arguments,
    )
except Exception:
    state_after_error = load_json(
        DATA_FILE
    ).get(CASE_ID, {})
    recorded_after_error = (
        isinstance(state_after_error, dict)
        and state_after_error.get(
            "human_decision",
            {},
        ).get("decision_id")
        == DECISION_ID
    )

    if fresh_decision and recorded_after_error:
        case_state._atomic_write_json(
            DATA_FILE,
            backup_cases,
        )
        print(
            "ROLLBACK AFTER WRITE ERROR: EXECUTED"
        )

    raise

after_cases = load_json(DATA_FILE)
after_case = after_cases.get(CASE_ID)

if not isinstance(after_case, dict):
    raise RuntimeError(
        "CASE-001 disappeared after decision recording"
    )

human_record = after_case.get(
    "human_decision"
)
current_submission = submission(CASE_ID)
current_compliance = compliance_rules(CASE_ID)

backup_version = backup_case.get(
    "state_version",
    0,
)

state_verified = all(
    (
        write_result.get("status")
        in {"RECORDED", "ALREADY_RECORDED"},
        isinstance(human_record, dict),
        human_record.get("decision_id")
        == DECISION_ID,
        human_record.get("decision")
        == HUMAN_DECISION,
        human_record.get("reason")
        == HUMAN_REASON,
        human_record.get("human_actor")
        == HUMAN_ACTOR,
        human_record.get("human_confirmation")
        is True,
        human_record.get("source")
        == DECISION_SOURCE,
        human_record.get("ai_recommendation")
        == AI_RECOMMENDATION,
        human_record.get(
            "ai_recommendation_followed"
        )
        is True,
        human_record.get("decision_scope")
        == "SYNTHETIC_TRAINING_ONLY",
        human_record.get("policy_next_step")
        == "HUMAN_REVIEW",
        human_record.get("review_gate_status")
        == "PERMITTED",
        human_record.get("authorization_status")
        == "HUMAN_AUTHORIZED_DECISION",
        after_case.get("workflow_status")
        == "REFERRED_FOR_SPECIALIST_REVIEW",
        after_case.get("coverage_decision_status")
        == "NOT_MADE",
        after_case.get("state_version")
        == backup_version + 1,
        after_case.get(
            "external_action_executed"
        )
        is False,
        after_case.get(
            "coverage_decision_executed"
        )
        is False,
        current_submission.get("evidence_status")
        == "COMPLETE",
        current_submission.get(
            "outstanding_evidence"
        )
        == [],
        current_compliance.get("next_step")
        == "HUMAN_REVIEW",
        file_hash(BACKUP_FILE)
        == backup_hash_before,
    )
)

if not state_verified:
    if write_result.get("status") == "RECORDED":
        case_state._atomic_write_json(
            DATA_FILE,
            backup_cases,
        )
        print(
            "STATE VALIDATION ROLLBACK: EXECUTED"
        )

    raise RuntimeError(
        "Persisted human-decision validation failed"
    )

telemetry_status = "ALREADY_EXISTS"

if not telemetry_event_exists(
    TELEMETRY_EVENT_ID
):
    record_event(
        {
            "event_id": TELEMETRY_EVENT_ID,
            "lab": "V2_CAPSTONE",
            "event_type": (
                "HUMAN_DECISION_RECORDED"
            ),
            "case_id": CASE_ID,
            "decision_id": DECISION_ID,
            "route": "HUMAN_REVIEW",
            "model_id": "NO_MODEL_CALL",
            "ai_recommendation": (
                AI_RECOMMENDATION
            ),
            "human_decision": HUMAN_DECISION,
            "ai_recommendation_followed": True,
            "authorization_status": (
                "HUMAN_AUTHORIZED_DECISION"
            ),
            "workflow_status": (
                "REFERRED_FOR_SPECIALIST_REVIEW"
            ),
            "model_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "estimated_model_cost_usd": 0.0,
            "external_action_executed": False,
            "coverage_decision_executed": False,
            "outcome_valid": True,
            "event_time": human_record.get(
                "decided_at"
            ),
        }
    )
    telemetry_status = "RECORDED"

telemetry_verified = telemetry_event_exists(
    TELEMETRY_EVENT_ID
)

report = {
    "case_id": CASE_ID,
    "decision_id": DECISION_ID,
    "write_status": write_result.get("status"),
    "previous_state_version": backup_version,
    "current_state_version": after_case.get(
        "state_version"
    ),
    "evidence_status": current_submission.get(
        "evidence_status"
    ),
    "policy_next_step": current_compliance.get(
        "next_step"
    ),
    "ai_recommendation": AI_RECOMMENDATION,
    "human_decision": HUMAN_DECISION,
    "human_reason": HUMAN_REASON,
    "ai_recommendation_followed": True,
    "authorization_status": human_record.get(
        "authorization_status"
    ),
    "workflow_status": after_case.get(
        "workflow_status"
    ),
    "coverage_decision_status": after_case.get(
        "coverage_decision_status"
    ),
    "external_action_executed": False,
    "coverage_decision_executed": False,
    "rollback_file": BACKUP_FILE.name,
    "rollback_verified": (
        file_hash(BACKUP_FILE)
        == backup_hash_before
    ),
    "telemetry_event_id": TELEMETRY_EVENT_ID,
    "telemetry_status": telemetry_status,
    "telemetry_verified": telemetry_verified,
    "human_decision_verified": (
        state_verified and telemetry_verified
    ),
}

write_json_atomic(
    RESULT_FILE,
    report,
)

print("HUMAN DECISION WRITE RESULT:")
print(
    json.dumps(
        {
            "status": write_result.get(
                "status"
            ),
            "case_id": CASE_ID,
            "decision_id": DECISION_ID,
            "decision": HUMAN_DECISION,
        },
        indent=2,
    )
)

print("\nAUTHORITATIVE DECISION STATE:")
print(
    json.dumps(
        {
            "previous_state_version": (
                backup_version
            ),
            "current_state_version": (
                after_case.get("state_version")
            ),
            "evidence_status": (
                current_submission.get(
                    "evidence_status"
                )
            ),
            "policy_next_step": (
                current_compliance.get(
                    "next_step"
                )
            ),
            "human_decision": (
                human_record.get("decision")
            ),
            "workflow_status": (
                after_case.get(
                    "workflow_status"
                )
            ),
            "coverage_decision_status": (
                after_case.get(
                    "coverage_decision_status"
                )
            ),
            "external_action_executed": (
                after_case.get(
                    "external_action_executed"
                )
            ),
            "coverage_decision_executed": (
                after_case.get(
                    "coverage_decision_executed"
                )
            ),
        },
        indent=2,
    )
)

print(
    "\nHUMAN DECISION STATE VERIFIED:",
    state_verified,
)
print(
    "LOCAL TELEMETRY VERIFIED:",
    telemetry_verified,
)
print(
    "ROLLBACK FILE UNCHANGED:",
    file_hash(BACKUP_FILE)
    == backup_hash_before,
)
print(
    "AUTHORITATIVE STATE CHANGES:",
    1
    if write_result.get("status") == "RECORDED"
    else 0,
)
print("MODEL CALLS EXECUTED: 0")
print("DATABRICKS CHANGES: 0")
print("EXTERNAL ACTIONS EXECUTED: 0")
print("COVERAGE DECISIONS EXECUTED: 0")
print("RESULT FILE:", RESULT_FILE.name)
print("SECRETS PRINTED: False")

if not (
    state_verified
    and telemetry_verified
):
    raise RuntimeError(
        "Human-decision persistence did not verify"
    )

print(
    "CAPSTONE HUMAN DECISION PERSISTENCE: COMPLETE"
)
