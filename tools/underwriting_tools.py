import json
from pathlib import Path

DATA_FILE = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "underwriting_cases.json"
)


def submission(case_id: str) -> dict:
    if not isinstance(case_id, str) or not case_id.strip():
        return {"error": "case_id must be a non-empty string"}

    normalized_id = case_id.strip().upper()

    # Read the current case data each time the tool runs.
    cases = json.loads(DATA_FILE.read_text(encoding="utf-8"))
    case = cases.get(normalized_id)

    if case is None:
        return {"error": f"Case not found: {normalized_id}"}

    outstanding = []
    for name in case["required_evidence"]:
        evidence = case["evidence"].get(name, {})
        if (
            evidence.get("received") is not True
            or evidence.get("verified") is not True
        ):
            outstanding.append(name)

    case["outstanding_evidence"] = outstanding
    case["evidence_status"] = "INCOMPLETE" if outstanding else "COMPLETE"
    return case


# Invented point rules for demonstrating deterministic scoring.
_TRAINING_POINTS = {
    "coastal_exposure": {"LOW": 0, "MEDIUM": 10, "HIGH": 25},
    "hurricane_exposure": {"LOW": 0, "MEDIUM": 15, "HIGH": 30},
    "flood_zone": {"AE": 20},
}


def _normalize_text(value):
    return value.strip().upper() if isinstance(value, str) else ""


def weather_risk(location: str) -> dict:
    location_id = _normalize_text(location)
    if not location_id:
        return {"error": "location must be a non-empty string"}

    cases = json.loads(DATA_FILE.read_text(encoding="utf-8"))
    matches = [
        case for case in cases.values()
        if _normalize_text(case.get("location")) == location_id
    ]

    if len(matches) != 1:
        return {"error": "Unknown or ambiguous training location"}

    case = matches[0]
    profile = {
        field: _normalize_text(case.get(field))
        for field in _TRAINING_POINTS
    }

    if not all(profile.values()):
        return {"error": "Exposure fields must contain non-empty text"}

    return {
        "location": location_id,
        "source": "SYNTHETIC_LAB_DATA",
        **profile,
    }


def compliance_rules(case_id: str) -> dict:
    case = submission(case_id)
    if "error" in case:
        return case

    return {
        "case_id": case["case_id"],
        "policy_id": "LAB-UW-POLICY-001",
        "required_evidence": case["required_evidence"],
        "outstanding_evidence": case["outstanding_evidence"],
        "evidence_status": case["evidence_status"],
        "human_review_required": True,
        "human_approval_required": True,
        "next_step": (
            "REQUEST_EVIDENCE"
            if case["outstanding_evidence"]
            else "HUMAN_REVIEW"
        ),
    }


def loss_history(case_id: str) -> dict:
    case = submission(case_id)
    if "error" in case:
        return case

    evidence = case["evidence"].get("contractor_loss_history", {})
    received = evidence.get("received") is True
    verified = received and evidence.get("verified") is True

    status = (
        "VERIFIED"
        if verified
        else "RECEIVED_UNVERIFIED"
        if received
        else "MISSING"
    )

    records = case.get("loss_history_records") if verified else None

    if verified and not isinstance(records, list):
        return {
            "error": (
                "Evidence is marked verified but "
                "loss records are unavailable"
            ),
            "case_id": case["case_id"],
        }

    return {
        "case_id": case["case_id"],
        "evidence_status": status,
        "received": received,
        "verified": verified,
        "records": records,
    }


def base_score(case_id: str) -> dict:
    # Read stored facts instead of accepting model-generated exposures.
    case = submission(case_id)
    if "error" in case:
        return case

    breakdown = []

    for field, point_table in _TRAINING_POINTS.items():
        value = _normalize_text(case.get(field))

        if value not in point_table:
            return {
                "error": (
                    f"Unsupported {field} for training score: {value}"
                )
            }

        breakdown.append({
            "factor": field,
            "value": value,
            "points": point_table[value],
        })

    return {
        "case_id": case["case_id"],
        "rule_set": "LAB-UW-SCORE-001",
        "purpose": "Illustrative training score",
        "score": sum(item["points"] for item in breakdown),
        "unit": "training_points",
        "breakdown": breakdown,
    }
