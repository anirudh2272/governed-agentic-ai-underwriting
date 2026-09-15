"""Shared persistent case-state management for the V2 capstone.

The JSON file simulates an authoritative store shared by the main
application and separate MCP server processes.
"""

from collections.abc import Mapping
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

from tools.underwriting_tools import DATA_FILE


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

CAPSTONE_BACKUP_FILE = (
    PROJECT_ROOT
    / "v2"
    / "data"
    / "capstone_underwriting_cases_backup.json"
)


def _utc_timestamp() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def _normalize_case_id(case_id: str) -> str:
    if not isinstance(case_id, str):
        raise TypeError("case_id must be a string.")

    normalized = case_id.strip().upper()

    if not normalized:
        raise ValueError("case_id cannot be blank.")

    return normalized


def _load_store(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"State file does not exist: {path}"
        )

    data = json.loads(
        path.read_text(encoding="utf-8")
    )

    if not isinstance(data, dict):
        raise ValueError(
            "Authoritative state must be a JSON object."
        )

    return data


def _atomic_write_json(
    path: Path,
    payload: Mapping[str, Any],
) -> None:
    """Atomically replace a JSON state file."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    file_descriptor, temporary_name = (
        tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
        )
    )
    temporary_path = Path(temporary_name)

    try:
        with os.fdopen(
            file_descriptor,
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(
                payload,
                handle,
                indent=2,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())

        if path.exists():
            temporary_path.chmod(
                path.stat().st_mode & 0o777
            )

        os.replace(
            temporary_path,
            path,
        )
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def read_shared_case(
    case_id: str,
) -> dict[str, Any]:
    """Read the latest case directly from shared storage."""

    normalized = _normalize_case_id(case_id)
    cases = _load_store(DATA_FILE)
    case = cases.get(normalized)

    if not isinstance(case, dict):
        raise KeyError(
            f"Case not found: {normalized}"
        )

    return case


def create_capstone_backup() -> dict[str, Any]:
    """Create one non-overwriting backup of pre-capstone state."""

    CAPSTONE_BACKUP_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if CAPSTONE_BACKUP_FILE.exists():
        status = "EXISTING_BACKUP_PRESERVED"
    else:
        shutil.copy2(
            DATA_FILE,
            CAPSTONE_BACKUP_FILE,
        )
        status = "BACKUP_CREATED"

    return {
        "status": status,
        "source": str(DATA_FILE),
        "backup": str(CAPSTONE_BACKUP_FILE),
    }


def restore_capstone_backup() -> dict[str, Any]:
    """Atomically restore the authoritative pre-capstone state."""

    backup_state = _load_store(
        CAPSTONE_BACKUP_FILE
    )
    _atomic_write_json(
        DATA_FILE,
        backup_state,
    )

    return {
        "status": "BACKUP_RESTORED",
        "source": str(CAPSTONE_BACKUP_FILE),
        "target": str(DATA_FILE),
    }


def apply_evidence_updates(
    case_id: str,
    updates: Mapping[str, Mapping[str, Any]],
    *,
    source: str = "SYNTHETIC_CAPSTONE",
) -> dict[str, Any]:
    """Apply validated evidence changes in one atomic transaction."""

    normalized = _normalize_case_id(case_id)

    if not isinstance(updates, Mapping) or not updates:
        raise ValueError(
            "At least one evidence update is required."
        )

    cases = _load_store(DATA_FILE)
    case = cases.get(normalized)

    if not isinstance(case, dict):
        raise KeyError(
            f"Case not found: {normalized}"
        )

    evidence = case.get("evidence")
    required_evidence = set(
        case.get("required_evidence", [])
    )

    if not isinstance(evidence, dict):
        raise ValueError(
            "Case evidence must be a JSON object."
        )

    unknown_evidence = (
        set(updates) - required_evidence
    )

    if unknown_evidence:
        raise ValueError(
            "Unknown evidence item(s): "
            + ", ".join(
                sorted(unknown_evidence)
            )
        )

    timestamp = _utc_timestamp()

    for evidence_name, patch in updates.items():
        if not isinstance(patch, Mapping):
            raise TypeError(
                f"{evidence_name} update must "
                "be an object."
            )

        received = patch.get("received")
        verified = patch.get("verified")

        if type(received) is not bool:
            raise ValueError(
                f"{evidence_name}.received "
                "must be Boolean."
            )

        if type(verified) is not bool:
            raise ValueError(
                f"{evidence_name}.verified "
                "must be Boolean."
            )

        if verified and not received:
            raise ValueError(
                f"{evidence_name} cannot be "
                "verified before it is received."
            )

        existing = evidence.get(
            evidence_name,
            {},
        )

        if not isinstance(existing, dict):
            raise ValueError(
                f"{evidence_name} state is invalid."
            )

        existing.update(dict(patch))
        existing["source"] = source
        existing["updated_at"] = timestamp

        evidence[evidence_name] = existing

    case["evidence"] = evidence
    case["state_version"] = (
        int(case.get("state_version", 0)) + 1
    )
    case["state_updated_at"] = timestamp
    cases[normalized] = case

    _atomic_write_json(
        DATA_FILE,
        cases,
    )

    return read_shared_case(normalized)


def apply_verified_evidence_package(
    package: Mapping[str, Any],
) -> dict[str, Any]:
    """Commit verified evidence and its records atomically."""

    if not isinstance(package, Mapping):
        raise TypeError(
            "Evidence package must be an object."
        )

    normalized = _normalize_case_id(
        package.get("case_id")
    )

    source = package.get("source")

    if not isinstance(source, str) or not source.strip():
        raise ValueError(
            "Evidence package source is required."
        )

    evidence_updates = package.get("evidence")
    loss_records = package.get(
        "loss_history_records"
    )

    if not isinstance(evidence_updates, Mapping):
        raise ValueError(
            "Evidence package must contain evidence."
        )

    if not isinstance(loss_records, list):
        raise ValueError(
            "loss_history_records must be a list."
        )

    if not loss_records:
        raise ValueError(
            "At least one loss-history record "
            "is required."
        )

    cases = _load_store(DATA_FILE)
    case = cases.get(normalized)

    if not isinstance(case, dict):
        raise KeyError(
            f"Case not found: {normalized}"
        )

    required_evidence = set(
        case.get("required_evidence", [])
    )
    supplied_evidence = set(evidence_updates)

    if supplied_evidence != required_evidence:
        missing = (
            required_evidence - supplied_evidence
        )
        unexpected = (
            supplied_evidence - required_evidence
        )

        raise ValueError(
            "Evidence package does not match "
            f"requirements. Missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}"
        )

    current_evidence = case.get("evidence")

    if not isinstance(current_evidence, dict):
        raise ValueError(
            "Current case evidence is invalid."
        )

    timestamp = _utc_timestamp()

    for evidence_name in sorted(
        required_evidence
    ):
        patch = evidence_updates[evidence_name]

        if not isinstance(patch, Mapping):
            raise TypeError(
                f"{evidence_name} must be an object."
            )

        if patch.get("received") is not True:
            raise ValueError(
                f"{evidence_name} is not received."
            )

        if patch.get("verified") is not True:
            raise ValueError(
                f"{evidence_name} is not verified."
            )

        current = current_evidence.get(
            evidence_name
        )

        if not isinstance(current, dict):
            raise ValueError(
                f"{evidence_name} current state "
                "is invalid."
            )

        current.update(dict(patch))
        current["source"] = source.strip()
        current["updated_at"] = timestamp

        current_evidence[evidence_name] = (
            current
        )

    required_record_fields = {
        "record_id",
        "loss_date",
        "loss_type",
        "incurred_loss_usd",
        "status",
        "source",
    }

    validated_records = []
    record_ids = set()

    for index, record in enumerate(
        loss_records,
        start=1,
    ):
        if not isinstance(record, Mapping):
            raise TypeError(
                f"Loss record {index} must "
                "be an object."
            )

        missing_fields = (
            required_record_fields - set(record)
        )

        if missing_fields:
            raise ValueError(
                f"Loss record {index} is missing "
                f"{sorted(missing_fields)}"
            )

        record_id = record["record_id"]

        if (
            not isinstance(record_id, str)
            or not record_id.strip()
        ):
            raise ValueError(
                f"Loss record {index} has an "
                "invalid record_id."
            )

        if record_id in record_ids:
            raise ValueError(
                f"Duplicate loss record: "
                f"{record_id}"
            )

        incurred = record[
            "incurred_loss_usd"
        ]

        if (
            isinstance(incurred, bool)
            or not isinstance(
                incurred,
                (int, float),
            )
            or incurred < 0
        ):
            raise ValueError(
                f"Loss record {record_id} has an "
                "invalid incurred amount."
            )

        record_ids.add(record_id)
        validated_records.append(dict(record))

    case["evidence"] = current_evidence
    case["loss_history_records"] = (
        validated_records
    )
    case["state_version"] = (
        int(case.get("state_version", 0)) + 1
    )
    case["state_updated_at"] = timestamp
    case["state_update_source"] = source.strip()

    cases[normalized] = case

    # Evidence flags and supporting records are written together.
    _atomic_write_json(
        DATA_FILE,
        cases,
    )

    return read_shared_case(normalized)

def record_human_decision(
    case_id: str,
    *,
    decision_id: str,
    decision: str,
    reason: str,
    human_actor: str,
    human_confirmation: bool,
    source: str,
    ai_recommendation: str,
) -> dict:
    # Record an explicit human decision without executing
    # any external or coverage action.
    import json as _json
    from datetime import datetime as _datetime
    from datetime import timezone as _timezone

    from tools.underwriting_controls import (
        authorize_underwriting_step as _authorize_step,
    )
    from tools.underwriting_tools import (
        compliance_rules as _compliance_rules,
    )

    def _required_text(value, field_name):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"{field_name} must be a non-empty string"
            )
        return value.strip()

    normalized_case_id = _required_text(
        case_id,
        "case_id",
    ).upper()
    normalized_decision_id = _required_text(
        decision_id,
        "decision_id",
    )
    normalized_decision = _required_text(
        decision,
        "decision",
    ).upper()
    normalized_reason = _required_text(
        reason,
        "reason",
    )
    normalized_actor = _required_text(
        human_actor,
        "human_actor",
    )
    normalized_source = _required_text(
        source,
        "source",
    ).upper()
    normalized_ai_recommendation = _required_text(
        ai_recommendation,
        "ai_recommendation",
    ).upper()

    valid_decisions = {
        "APPROVE",
        "DECLINE",
        "REFER",
    }

    if normalized_decision not in valid_decisions:
        raise ValueError(
            "decision must be APPROVE, DECLINE, or REFER"
        )

    if (
        normalized_ai_recommendation
        not in valid_decisions
    ):
        raise ValueError(
            "ai_recommendation must be "
            "APPROVE, DECLINE, or REFER"
        )

    if human_confirmation is not True:
        raise PermissionError(
            "Explicit human confirmation is required"
        )

    if normalized_source != "EXPLICIT_USER_INPUT":
        raise PermissionError(
            "Decision source must be EXPLICIT_USER_INPUT"
        )

    cases = _json.loads(
        DATA_FILE.read_text(encoding="utf-8")
    )
    case = cases.get(normalized_case_id)

    if not isinstance(case, dict):
        raise KeyError(
            f"Case not found: {normalized_case_id}"
        )

    if (
        case.get("external_action_executed") is True
        or case.get("coverage_decision_executed")
        is True
    ):
        raise RuntimeError(
            "The case reports a previously executed action"
        )

    required_evidence = case.get(
        "required_evidence"
    )
    evidence = case.get("evidence")

    if (
        not isinstance(required_evidence, list)
        or not required_evidence
        or not isinstance(evidence, dict)
    ):
        raise RuntimeError(
            "Required evidence state is invalid"
        )

    incomplete_evidence = [
        name
        for name in required_evidence
        if (
            not isinstance(evidence.get(name), dict)
            or evidence[name].get("received") is not True
            or evidence[name].get("verified") is not True
        )
    ]

    if incomplete_evidence:
        raise PermissionError(
            "Human decision cannot be recorded while "
            "evidence is incomplete: "
            + ", ".join(incomplete_evidence)
        )

    if "contractor_loss_history" in required_evidence:
        loss_records = case.get(
            "loss_history_records"
        )

        if (
            not isinstance(loss_records, list)
            or not loss_records
            or not all(
                isinstance(record, dict)
                for record in loss_records
            )
        ):
            raise RuntimeError(
                "Verified contractor loss-history "
                "records are unavailable"
            )

    compliance = _compliance_rules(
        normalized_case_id
    )

    if (
        not isinstance(compliance, dict)
        or "error" in compliance
        or compliance.get("evidence_status")
        != "COMPLETE"
        or compliance.get("next_step")
        != "HUMAN_REVIEW"
    ):
        raise PermissionError(
            "Current policy state does not permit "
            "human-review decision recording"
        )

    review_gate = _authorize_step(
        normalized_case_id,
        "HUMAN_REVIEW",
        compliance,
        human_approval_supplied=False,
    )

    if (
        review_gate.get("authorization_status")
        != "PERMITTED"
    ):
        raise PermissionError(
            "Application review gate blocked the "
            "human decision: "
            + str(review_gate.get("reason_code"))
        )

    requested_identity = {
        "decision_id": normalized_decision_id,
        "decision": normalized_decision,
        "reason": normalized_reason,
        "human_actor": normalized_actor,
        "human_confirmation": True,
        "source": normalized_source,
        "ai_recommendation": (
            normalized_ai_recommendation
        ),
        "decision_scope": (
            "SYNTHETIC_TRAINING_ONLY"
        ),
    }

    existing_decision = case.get(
        "human_decision"
    )

    if existing_decision is not None:
        if not isinstance(existing_decision, dict):
            raise RuntimeError(
                "Existing human decision is malformed"
            )

        same_decision = all(
            existing_decision.get(key) == value
            for key, value
            in requested_identity.items()
        )

        if not same_decision:
            raise RuntimeError(
                "Conflicting human decision overwrite "
                "was blocked"
            )

        return {
            "status": "ALREADY_RECORDED",
            "case_id": normalized_case_id,
            "human_decision": dict(
                existing_decision
            ),
            "case": read_shared_case(
                normalized_case_id
            ),
        }

    current_version = case.get(
        "state_version",
        0,
    )

    if (
        isinstance(current_version, bool)
        or not isinstance(current_version, int)
        or current_version < 0
    ):
        raise RuntimeError(
            "state_version must be a non-negative integer"
        )

    timestamp = _datetime.now(
        _timezone.utc
    ).isoformat()

    workflow_statuses = {
        "APPROVE": "HUMAN_REVIEW_APPROVED",
        "DECLINE": "HUMAN_REVIEW_DECLINED",
        "REFER": "REFERRED_FOR_SPECIALIST_REVIEW",
    }

    human_decision_record = {
        **requested_identity,
        "ai_recommendation_followed": (
            normalized_decision
            == normalized_ai_recommendation
        ),
        "policy_id": compliance.get(
            "policy_id"
        ),
        "policy_next_step": compliance.get(
            "next_step"
        ),
        "review_gate_status": review_gate.get(
            "authorization_status"
        ),
        "review_gate_reason_code": review_gate.get(
            "reason_code"
        ),
        "authorization_status": (
            "HUMAN_AUTHORIZED_DECISION"
        ),
        "decided_at": timestamp,
    }

    case["human_decision"] = (
        human_decision_record
    )
    case["workflow_status"] = (
        workflow_statuses[normalized_decision]
    )
    case["coverage_decision_status"] = "NOT_MADE"
    case["external_action_executed"] = False
    case["coverage_decision_executed"] = False
    case["state_version"] = current_version + 1
    case["state_updated_at"] = timestamp
    case["state_update_source"] = (
        normalized_source
    )

    cases[normalized_case_id] = case

    _atomic_write_json(
        DATA_FILE,
        cases,
    )

    return {
        "status": "RECORDED",
        "case_id": normalized_case_id,
        "human_decision": dict(
            human_decision_record
        ),
        "case": read_shared_case(
            normalized_case_id
        ),
    }
