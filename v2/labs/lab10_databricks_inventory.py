import os
from pathlib import Path

from databricks import sql
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def quote_identifier(value: str) -> str:
    return "`" + value.replace("`", "``") + "`"


def probe_table(
    cursor,
    label: str,
    table_name: str,
) -> str:
    try:
        cursor.execute(
            f"SELECT * FROM {table_name} LIMIT 0"
        )
        return "ACCESSIBLE"
    except Exception as error:
        return (
            "NOT_ACCESSIBLE "
            f"({type(error).__name__})"
        )


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
        "DATABRICKS_AUTH_TYPE must be "
        "databricks-oauth."
    )

print("OPENING DATABRICKS CONNECTION...")

with sql.connect(
    server_hostname=hostname,
    http_path=http_path,
    auth_type=auth_type,
) as connection:
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT
                current_user(),
                current_catalog(),
                current_schema()
            """
        )
        session = cursor.fetchone()

        current_user = session[0]
        current_catalog = session[1]
        current_schema = session[2]

        print("\nSESSION:")
        print("CURRENT USER:", current_user)
        print(
            "CURRENT CATALOG:",
            current_catalog,
        )
        print(
            "CURRENT SCHEMA:",
            current_schema,
        )

        cursor.execute("SHOW CATALOGS")
        catalog_rows = cursor.fetchall()
        catalogs = sorted(
            str(row[0])
            for row in catalog_rows
        )

        print("\nACCESSIBLE CATALOGS:")
        for catalog in catalogs:
            print("-", catalog)

        print("\nSCHEMA INVENTORY:")

        schema_inventory = {}

        for catalog in catalogs:
            try:
                cursor.execute(
                    "SHOW SCHEMAS IN "
                    + quote_identifier(catalog)
                )
                schema_rows = cursor.fetchall()
                schema_names = sorted(
                    str(row[0])
                    for row in schema_rows
                )
                schema_inventory[
                    catalog
                ] = schema_names

                preview = schema_names[:20]

                print(
                    f"- {catalog}: "
                    f"{len(schema_names)} schema(s)"
                )

                for schema_name in preview:
                    print(
                        "    ",
                        schema_name,
                    )

                if len(schema_names) > 20:
                    print("     ...")
            except Exception as error:
                schema_inventory[catalog] = []
                print(
                    f"- {catalog}: "
                    "NOT_ACCESSIBLE "
                    f"({type(error).__name__})"
                )

        system_table_status = {
            "billing_usage": probe_table(
                cursor,
                "billing_usage",
                "system.billing.usage",
            ),
            "billing_list_prices": probe_table(
                cursor,
                "billing_list_prices",
                "system.billing.list_prices",
            ),
            "ai_gateway_usage": probe_table(
                cursor,
                "ai_gateway_usage",
                "system.ai_gateway.usage",
            ),
            "external_model_spend": probe_table(
                cursor,
                "external_model_spend",
                (
                    "system.ai_gateway."
                    "external_model_spend"
                ),
            ),
        }

        print("\nSYSTEM TABLE ACCESS:")
        for name, status in (
            system_table_status.items()
        ):
            print(f"- {name}: {status}")

        writable_candidates = [
            catalog
            for catalog in catalogs
            if catalog.lower()
            not in {
                "samples",
                "system",
                "__databricks_internal",
            }
        ]

        inventory_valid = all(
            (
                bool(current_user),
                bool(current_catalog),
                bool(catalogs),
                current_catalog in catalogs,
            )
        )

        print(
            "\nUSER CATALOG CANDIDATES:",
            writable_candidates,
        )
        print(
            "INVENTORY VALID:",
            inventory_valid,
        )
        print("CHANGES EXECUTED: 0")
        print("SECRETS PRINTED: False")

        if not inventory_valid:
            raise RuntimeError(
                "Databricks inventory validation "
                "failed."
            )

print("LAB 10 DISCOVERY STATUS: COMPLETE")
