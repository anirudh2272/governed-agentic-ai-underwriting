import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TELEMETRY_FILE = (
    PROJECT_ROOT
    / "v2"
    / "data"
    / "interaction_telemetry.jsonl"
)


def record_event(event: dict[str, Any]) -> None:
    """Append one model interaction as a JSON Lines record."""

    TELEMETRY_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with TELEMETRY_FILE.open(
        "a",
        encoding="utf-8",
    ) as file:
        file.write(
            json.dumps(event, sort_keys=True) + "\n"
        )
