# Fictional customer data for this lab.
# typical_max describes usual behavior, not an approval limit.

CUSTOMERS = {
    "C1001": {
        "customer_id": "C1001",
        "risk": "MEDIUM",
        "typical_max": 2000,
        "new_device_alert": True,
    }
}


def get_customer(customer_id: str) -> dict:
    return CUSTOMERS.get(
        customer_id,
        {"error": "Customer not found"},
    )


# Additional fictional data for Lab 4.
TRANSACTIONS = {
    "T1001": {
        "transaction_id": "T1001",
        "customer_id": "C1001",
        "amount": 25000,
        "currency": "USD",
        "device_id": "D1001",
    }
}

DEVICES = {
    "D1001": {
        "device_id": "D1001",
        "customer_id": "C1001",
        "is_new": True,
        "verification_status": "UNVERIFIED",
    }
}


def get_transaction(transaction_id: str) -> dict:
    return TRANSACTIONS.get(
        transaction_id,
        {"error": "Transaction not found"},
    )


def get_device_status(device_id: str) -> dict:
    return DEVICES.get(
        device_id,
        {"error": "Device not found"},
    )


def get_risk_rules() -> dict:
    return {
        "policy_id": "LAB-RULES-001",
        "purpose": "Fictional training policy",
        "rules": [
            {
                "rule_id": "R1",
                "condition": "amount > 5 * typical_max",
                "recommendation": "HUMAN_REVIEW",
            },
            {
                "rule_id": "R2",
                "condition": "Transaction device is UNVERIFIED",
                "recommendation": "HUMAN_REVIEW",
            },
        ],
        "combination": "Either matching rule recommends human review",
    }


def approve_transaction(transaction_id: str) -> dict:
    # This training function always blocks the approval action.
    return {
        "transaction_id": transaction_id,
        "executed": False,
        "reason": "Human approval required",
    }
