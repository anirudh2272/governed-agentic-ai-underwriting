"""Capstone Phase A and shared-state transition."""

import json
from hashlib import sha256
from pathlib import Path
import subprocess
import sys
from typing import Any

from tools.underwriting_tools import (
    DATA_FILE,
    compliance_rules,
    loss_history,
    submission,
)
from v2.control_plane.enterprise_control_plane import (
    run_control_plane,
)
from v2.control_plane.model_selector import (
    load_model_catalog,
)
from v2.services.case_state import (
    CAPSTONE_BACKUP_FILE,
    apply_verified_evidence_package,
    restore_capstone_backup,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

PACKAGE_FILE = (
    PROJECT_ROOT
    / "v2"
    / "data"
    / "capstone_evidence_package.json"
)

RESULT_FILE = (
    PROJECT_ROOT
    / "v2"
    / "data"
    / "capstone_state_transition.json"
)


def file_hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def phase_a_summary(
    result: dict[str, Any],
) -> dict[str, Any]:
    selection = result.get("model_selection") or {}
    execution = result.get("execution") or {}
    authorization = result.get("authorization") or {}

    return {
        "route": result.get("route"),
        "control_plane_status": result.get(
            "control_plane_status"
        ),
        "model_id": selection.get("model_id"),
        "estimated_model_cost_usd": selection.get(
            "estimated_model_cost_usd"
        ),
        "model_calls": execution.get(
            "model_calls",
            0,
        ),
        "response_text": execution.get(
            "response_text"
        ),
        "authorization_status": authorization.get(
            "authorization_status"
        ),
        "external_action_executed": result.get(
            "external_action_executed"
        ),
        "coverage_decision_executed": result.get(
            "coverage_decision_executed"
        ),
    }


if not DATA_FILE.exists():
    raise SystemExit(
        "Authoritative state file is missing."
    )

if not CAPSTONE_BACKUP_FILE.exists():
    raise SystemExit(
        "Capstone backup is missing. "
        "Do not continue."
    )

if not PACKAGE_FILE.exists():
    raise SystemExit(
        "Evidence package is missing."
    )


authoritative_hash_before = file_hash(DATA_FILE)
backup_hash = file_hash(CAPSTONE_BACKUP_FILE)

if authoritative_hash_before != backup_hash:
    raise SystemExit(
        "Authoritative state no longer matches "
        "the pre-capstone backup. Do not continue."
    )


before_case = submission("CASE-001")
before_compliance = compliance_rules("CASE-001")

preconditions_valid = all(
    (
        before_case.get("evidence_status")
        == "INCOMPLETE",
        before_compliance.get("next_step")
        == "REQUEST_EVIDENCE",
    )
)

print("PRE-TRANSITION STATE:")
print(
    json.dumps(
        {
            "evidence_status": (
                before_case.get(
                    "evidence_status"
                )
            ),
            "outstanding_evidence": (
                before_case.get(
                    "outstanding_evidence"
                )
            ),
            "policy_next_step": (
                before_compliance.get(
                    "next_step"
                )
            ),
            "backup_matches_authoritative": (
                authoritative_hash_before
                == backup_hash
            ),
        },
        indent=2,
    )
)

if not preconditions_valid:
    raise SystemExit(
        "CASE-001 is not in the expected "
        "pre-capstone state."
    )


print("\nOPENING DATABRICKS MODEL CATALOG...")

catalog = load_model_catalog()


phase_a = run_control_plane(
    case_id="CASE-001",
    task_type="WORKFLOW_NEXT_STEP",
    deterministic_answer=(
        before_compliance["next_step"]
    ),
    complexity="LOW",
    business_risk="HIGH",
    data_classification="SYNTHETIC_TRAINING",
    requested_action="REQUEST_EVIDENCE",
    requested_tools=(),
    requested_agent_steps=0,
    expected_input_tokens=0,
    expected_output_tokens=0,
    catalog_entries=catalog,
)


phase_a_valid = all(
    (
        phase_a.get("control_plane_status")
        == "COMPLETED",
        phase_a.get("route") == "NO_LLM",
        phase_a["model_selection"].get(
            "model_id"
        )
        == "NO_LLM",
        phase_a["model_selection"].get(
            "estimated_model_cost_usd"
        )
        == 0.0,
        phase_a["execution"].get(
            "model_calls"
        )
        == 0,
        phase_a["execution"].get(
            "response_text"
        )
        == "REQUEST_EVIDENCE",
        phase_a["authorization"].get(
            "authorization_status"
        )
        == "PERMITTED",
        phase_a.get(
            "external_action_executed"
        )
        is False,
    )
)


print("\nCAPSTONE PHASE A:")
print(
    json.dumps(
        phase_a_summary(phase_a),
        indent=2,
    )
)

print(
    "PHASE A VERIFIED:",
    phase_a_valid,
)

if not phase_a_valid:
    raise SystemExit(
        "Phase A validation failed. "
        "Authoritative state was not changed."
    )


package = json.loads(
    PACKAGE_FILE.read_text(
        encoding="utf-8"
    )
)

state_mutated = False

try:
    committed_case = (
        apply_verified_evidence_package(
            package
        )
    )
    state_mutated = True

    after_case = submission("CASE-001")
    after_compliance = compliance_rules(
        "CASE-001"
    )
    after_losses = loss_history("CASE-001")

    subprocess_code = """
import json
from tools.underwriting_tools import (
    compliance_rules,
    loss_history,
    submission,
)

print(
    json.dumps(
        {
            "submission": submission("CASE-001"),
            "compliance": compliance_rules(
                "CASE-001"
            ),
            "loss_history": loss_history(
                "CASE-001"
            ),
        }
    )
)
"""

    subprocess_result = subprocess.run(
        [
            sys.executable,
            "-c",
            subprocess_code,
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    separate_process_state = json.loads(
        subprocess_result.stdout.strip()
    )

    separate_submission = (
        separate_process_state["submission"]
    )
    separate_compliance = (
        separate_process_state["compliance"]
    )
    separate_losses = (
        separate_process_state["loss_history"]
    )

    authoritative_hash_after = file_hash(
        DATA_FILE
    )

    local_state_valid = all(
        (
            committed_case.get(
                "state_version",
                0,
            )
            >= 1,
            after_case.get(
                "evidence_status"
            )
            == "COMPLETE",
            after_case.get(
                "outstanding_evidence"
            )
            == [],
            after_compliance.get("next_step")
            == "HUMAN_REVIEW",
            after_losses.get(
                "evidence_status"
            )
            == "VERIFIED",
            len(
                after_losses.get(
                    "records",
                    [],
                )
            )
            == 2,
        )
    )

    separate_process_valid = all(
        (
            separate_submission.get(
                "evidence_status"
            )
            == "COMPLETE",
            separate_submission.get(
                "outstanding_evidence"
            )
            == [],
            separate_compliance.get(
                "next_step"
            )
            == "HUMAN_REVIEW",
            separate_losses.get(
                "evidence_status"
            )
            == "VERIFIED",
            len(
                separate_losses.get(
                    "records",
                    [],
                )
            )
            == 2,
        )
    )

    state_changed = (
        authoritative_hash_after
        != authoritative_hash_before
    )

    transition_verified = all(
        (
            phase_a_valid,
            local_state_valid,
            separate_process_valid,
            state_changed,
        )
    )

    transition_report = {
        "phase_a": phase_a_summary(phase_a),
        "state_transition": {
            "before_evidence_status": (
                before_case.get(
                    "evidence_status"
                )
            ),
            "before_policy_next_step": (
                before_compliance.get(
                    "next_step"
                )
            ),
            "after_evidence_status": (
                after_case.get(
                    "evidence_status"
                )
            ),
            "after_policy_next_step": (
                after_compliance.get(
                    "next_step"
                )
            ),
            "loss_history_status": (
                after_losses.get(
                    "evidence_status"
                )
            ),
            "loss_record_count": len(
                after_losses.get(
                    "records",
                    [],
                )
            ),
            "state_version": (
                committed_case.get(
                    "state_version"
                )
            ),
            "state_changed": state_changed,
        },
        "local_state_valid": local_state_valid,
        "separate_process_valid": (
            separate_process_valid
        ),
        "transition_verified": (
            transition_verified
        ),
        "model_calls": 0,
        "estimated_model_cost_usd": 0.0,
        "external_actions_executed": 0,
    }

    if not transition_verified:
        raise RuntimeError(
            "Post-transition validation failed."
        )

    RESULT_FILE.write_text(
        json.dumps(
            transition_report,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

except Exception:
    if state_mutated:
        rollback = restore_capstone_backup()
        print("\nAUTOMATIC ROLLBACK:")
        print(json.dumps(rollback, indent=2))

    raise


print("\nAUTHORITATIVE STATE TRANSITION:")
print(
    json.dumps(
        transition_report[
            "state_transition"
        ],
        indent=2,
    )
)

print("\nSEPARATE PROCESS:")
print(
    json.dumps(
        {
            "evidence_status": (
                separate_submission.get(
                    "evidence_status"
                )
            ),
            "policy_next_step": (
                separate_compliance.get(
                    "next_step"
                )
            ),
            "loss_history_status": (
                separate_losses.get(
                    "evidence_status"
                )
            ),
            "loss_record_count": len(
                separate_losses.get(
                    "records",
                    [],
                )
            ),
        },
        indent=2,
    )
)

print(
    "\nLOCAL STATE VALID:",
    local_state_valid,
)
print(
    "SEPARATE PROCESS STATE VALID:",
    separate_process_valid,
)
print(
    "CAPSTONE STATE TRANSITION VERIFIED:",
    transition_verified,
)
print("CLAUDE API CALLS: 0")
print("DATABRICKS READS: 1")
print("DATABRICKS WRITES: 0")
print("UNDERWRITING ACTIONS EXECUTED: 0")
print(
    "ROLLBACK FILE:",
    CAPSTONE_BACKUP_FILE.name,
)
print("SECRETS PRINTED: False")
print("CAPSTONE PHASE A AND STATE TRANSITION: COMPLETE")
