"""Lightweight tests for playbook contract constants (no orchestrator tool package import)."""

import unittest

from orchestrator.utils.playbook_contracts import (
    DEEP_AGENT_SISTER_PHASE_OUTPUT_FIELDS,
    VALID_DEEP_AGENT_PHASE_TYPES,
    VALID_PLAYBOOK_STEP_TYPES,
)


class PlaybookContractsModuleTests(unittest.TestCase):
    def test_browser_authenticate_in_step_types(self) -> None:
        self.assertIn("browser_authenticate", VALID_PLAYBOOK_STEP_TYPES)

    def test_rerank_in_deep_phase_types(self) -> None:
        self.assertIn("rerank", VALID_DEEP_AGENT_PHASE_TYPES)

    def test_sister_phase_fields(self) -> None:
        self.assertIn("output", DEEP_AGENT_SISTER_PHASE_OUTPUT_FIELDS)
        self.assertIn("feedback", DEEP_AGENT_SISTER_PHASE_OUTPUT_FIELDS)


if __name__ == "__main__":
    unittest.main()
