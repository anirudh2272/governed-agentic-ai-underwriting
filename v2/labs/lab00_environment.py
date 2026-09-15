import os
import time
from pathlib import Path

from databricks import sql
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

hostname = os.getenv(
    "DATABRICKS_SERVER_HOSTNAME",
    "",
).strip()

http_path = os.getenv(
    "DATABRICKS_HTTP_PATH",
    "",
).strip()

auth_type = os.getenv(
    "DATABRICKS_AUTH_TYPE",
    "",
).strip()

if not hostname:
    raise SystemExit(
        "DATABRICKS_SERVER_HOSTNAME is missing."
    )

if not http_path:
    raise SystemExit(
        "DATABRICKS_HTTP_PATH is missing."
    )

if auth_type != "databricks-oauth":
    raise SystemExit(
        "Expected DATABRICKS_AUTH_TYPE="
        "databricks-oauth."
    )

print("LOCAL CONFIGURATION: VERIFIED")
print("AUTHENTICATION METHOD: OAUTH U2M")
print("OPENING DATABRICKS CONNECTION...")

started = time.perf_counter()

with sql.connect(
    server_hostname=hostname,
    http_path=http_path,
    auth_type=auth_type,
) as connection:
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT 1 AS connection_test"
        )
        row = cursor.fetchone()

elapsed = time.perf_counter() - started

if row is None or row[0] != 1:
    raise RuntimeError(
        f"Unexpected SQL result: {row}"
    )

print("SQL RESULT:", row[0])
print("CONNECTION TEST: SUCCESS")
print(
    "ELAPSED SECONDS:",
    round(elapsed, 3),
)
print("SECRETS PRINTED: False")
print("V2 LAB 0 STATUS: COMPLETE")
