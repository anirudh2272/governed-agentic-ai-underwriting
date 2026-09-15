import unittest

from tools.underwriting_controls import (
    authorize_underwriting_step,
)
from v2.control_plane.model_selector import (
    select_model,
)
from v2.control_plane.policies import (
    get_route_policy,
    is_data_classification_allowed,
)
from v2.control_plane.router import route_request
from v2.control_plane.tool_gateway import (
    DEFAULT_ALLOWED_TOOLS,
    execute_governed_tool,
)
from v2.services.route_executor import execute_route


EXPECTED_TOOLS = {
    "submission",
    "weather_risk",
    "compliance_rules",
    "loss_history",
    "base_score",
}


def qualified_conversational_entry():
    return {
        "catalog_entry_id": "offline-test-model",
        "route_name": "CONVERSATIONAL",
        "model_id": "offline-test-model",
        "provider": "TEST",
        "capability_tier": "CONVERSATIONAL",
        "governance_status": (
            "QUALIFIED_FOR_TRAINING"
        ),
        "active": True,
        "allowed_for_consequential_decisions": False,
        "input_cost_per_million": 1.0,
        "output_cost_per_million": 5.0,
        "input_cost_usd_per_million": 1.0,
        "output_cost_usd_per_million": 5.0,
        "qualification_sample_size": 3,
    }


class RouterTests(unittest.TestCase):
    def test_authoritative_answer_uses_no_llm(self):
        decision = route_request(
            task_type="WORKFLOW_NEXT_STEP",
            deterministic_answer="REQUEST_EVIDENCE",
            complexity="HIGH",
            business_risk="HIGH",
        )

        self.assertEqual(decision["route"], "NO_LLM")
        self.assertFalse(decision["model_required"])
        self.assertEqual(
            decision["deterministic_result"],
            "REQUEST_EVIDENCE",
        )

    def test_low_risk_summary_uses_conversational(self):
        decision = route_request(
            task_type="CASE_SUMMARY",
            deterministic_answer=None,
            complexity="LOW",
            business_risk="LOW",
        )

        self.assertEqual(
            decision["route"],
            "CONVERSATIONAL",
        )
        self.assertTrue(decision["model_required"])

    def test_high_risk_work_uses_premium(self):
        decision = route_request(
            task_type="UNDERWRITING_RECOMMENDATION",
            deterministic_answer=None,
            complexity="HIGH",
            business_risk="HIGH",
        )

        self.assertEqual(
            decision["route"],
            "PREMIUM_REASONING",
        )
        self.assertTrue(decision["model_required"])


class PolicyTests(unittest.TestCase):
    def test_expected_tool_allowlist(self):
        self.assertEqual(
            set(DEFAULT_ALLOWED_TOOLS),
            EXPECTED_TOOLS,
        )

    def test_restricted_data_policy(self):
        self.assertFalse(
            is_data_classification_allowed(
                "PREMIUM_REASONING",
                "RESTRICTED",
            )
        )
        self.assertTrue(
            is_data_classification_allowed(
                "NO_LLM",
                "RESTRICTED",
            )
        )

    def test_no_llm_has_zero_agent_steps(self):
        policy = get_route_policy("NO_LLM")

        self.assertEqual(policy["max_agent_steps"], 0)
        self.assertEqual(
            set(policy["allowed_tools"]),
            set(),
        )


class ModelSelectorTests(unittest.TestCase):
    def test_selects_qualified_entry(self):
        selection = select_model(
            "CONVERSATIONAL",
            [qualified_conversational_entry()],
            expected_input_tokens=500,
            expected_output_tokens=100,
        )

        self.assertEqual(
            selection["selection_status"],
            "SELECTED",
        )
        self.assertEqual(
            selection["model_id"],
            "offline-test-model",
        )
        self.assertFalse(
            selection[
                "allowed_for_consequential_decisions"
            ]
        )

    def test_unqualified_route_fails_closed(self):
        selection = select_model(
            "PREMIUM_REASONING",
            [],
        )

        self.assertEqual(
            selection["selection_status"],
            "NO_QUALIFIED_MODEL",
        )
        self.assertEqual(
            selection["reason_code"],
            "FAIL_CLOSED_CATALOG_FILTER",
        )
        self.assertFalse(
            selection["model_call_executed"]
        )

    def test_consequential_request_is_blocked(self):
        selection = select_model(
            "CONVERSATIONAL",
            [qualified_conversational_entry()],
            consequential_action_requested=True,
        )

        self.assertEqual(
            selection["selection_status"],
            "BLOCKED",
        )
        self.assertEqual(
            selection["reason_code"],
            (
                "CONSEQUENTIAL_ACTION_REQUIRES_"
                "APPLICATION_AUTHORIZATION"
            ),
        )


class ToolGatewayTests(unittest.TestCase):
    def test_unapproved_tool_is_blocked(self):
        result = execute_governed_tool(
            "bind_coverage",
            {},
        )

        self.assertEqual(result["status"], "BLOCKED")
        self.assertEqual(
            result["reason_code"],
            "TOOL_NOT_ALLOWED",
        )
        self.assertFalse(result["executed"])
        self.assertFalse(
            result["external_action_executed"]
        )

    def test_extra_argument_is_blocked(self):
        result = execute_governed_tool(
            "submission",
            {
                "case_id": "CASE-001",
                "unexpected": True,
            },
        )

        self.assertEqual(result["status"], "BLOCKED")
        self.assertEqual(
            result["reason_code"],
            "INVALID_ARGUMENTS",
        )
        self.assertFalse(result["executed"])


class AuthorizationTests(unittest.TestCase):
    def setUp(self):
        self.compliance = {
            "case_id": "CASE-001",
            "policy_id": "LAB-UW-POLICY-001",
            "next_step": "REQUEST_EVIDENCE",
            "human_review_required": True,
            "human_approval_required": True,
        }

    def test_policy_step_is_permitted(self):
        result = authorize_underwriting_step(
            "CASE-001",
            "REQUEST_EVIDENCE",
            self.compliance,
        )

        self.assertEqual(
            result["authorization_status"],
            "PERMITTED",
        )
        self.assertFalse(
            result["external_action_executed"]
        )
        self.assertFalse(
            result["coverage_decision_executed"]
        )

    def test_bind_coverage_requires_human_approval(self):
        result = authorize_underwriting_step(
            "CASE-001",
            "BIND_COVERAGE",
            self.compliance,
            human_approval_supplied=False,
        )

        self.assertEqual(
            result["authorization_status"],
            "BLOCKED",
        )
        self.assertEqual(
            result["reason_code"],
            "HUMAN_APPROVAL_REQUIRED",
        )
        self.assertFalse(
            result["external_action_executed"]
        )
        self.assertFalse(
            result["coverage_decision_executed"]
        )


class DeterministicExecutionTests(unittest.TestCase):
    def test_no_llm_execution_needs_no_client(self):
        decision = route_request(
            task_type="WORKFLOW_NEXT_STEP",
            deterministic_answer="REQUEST_EVIDENCE",
            complexity="HIGH",
            business_risk="HIGH",
        )

        result = execute_route(decision)

        self.assertEqual(result["route"], "NO_LLM")
        self.assertEqual(
            result["execution_mode"],
            "DETERMINISTIC",
        )
        self.assertEqual(result["model_calls"], 0)
        self.assertEqual(result["input_tokens"], 0)
        self.assertEqual(result["output_tokens"], 0)
        self.assertEqual(
            result["response_text"],
            "REQUEST_EVIDENCE",
        )


if __name__ == "__main__":
    unittest.main()
