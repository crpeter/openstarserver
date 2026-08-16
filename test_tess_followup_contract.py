import json
import tempfile
import unittest
from pathlib import Path

from workflows.tess.tess_followup import build_low_frequency_followup
from workflows.tess.tess_hypotheses import analyze, plan


def _identity():
    return {
        "queryErrors": [],
        "tic": {"found": True, "metadata": {}},
        "vsx": {"matches": []},
        "gaia": {},
        "gaiaVariability": {"periodCandidates": []},
    }


def _primary(preferred_period):
    return {
        "periodStatus": "RELIABLE",
        "periodConfidence": "high",
        "candidatePeriodDays": preferred_period / 2.0,
        "preferredPhysicalPeriodDays": preferred_period,
        "preferredPhysicalPeriodRelation": "2x",
        "harmonicCandidates": [],
    }


class LowFrequencyFollowupContractTests(unittest.TestCase):
    def test_blind_b_shaped_harmonic_uses_independent_sector(self):
        analysis = analyze(
            _primary(0.8001069097176371),
            _identity(),
            observation_baseline_days=27.879261016845703,
            primary_minimum_frequency=0.1,
        )

        decision = plan(analysis, _identity())

        self.assertFalse(analysis["harmonicLowFrequencyFollowup"]["applicable"])
        self.assertEqual("INDEPENDENT_SECTOR_FOLLOWUP", decision["action"])
        self.assertEqual("harmonic-cycle-needs-independent-sector", decision["reason"])

    def test_valid_harmonic_low_frequency_followup_is_preserved(self):
        analysis = analyze(
            _primary(20.0),
            _identity(),
            observation_baseline_days=40.0,
            primary_minimum_frequency=0.1,
        )

        decision = plan(analysis, _identity())

        self.assertTrue(analysis["harmonicLowFrequencyFollowup"]["applicable"])
        self.assertEqual("LOW_FREQUENCY_FOLLOWUP", decision["action"])
        self.assertEqual("harmonic-cycle-preferred", decision["reason"])

    def test_unavailable_preparation_returns_structured_outcome(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset_path = root / "dataset.json"
            project_path = root / "project.json"
            dataset_path.write_text(
                json.dumps(
                    {
                        "id": "generic-target",
                        "times": [0.0, 27.0],
                        "frequencySearch": {"minimumFrequency": 0.1},
                    }
                ),
                encoding="utf-8",
            )
            entry = {"id": "generic-target", "path": str(dataset_path)}
            project_path.write_text(
                json.dumps(
                    {
                        "id": "generic-project",
                        "name": "Generic",
                        "workloadID": "generic.period-search",
                        "datasets": [entry],
                    }
                ),
                encoding="utf-8",
            )

            result = build_low_frequency_followup(
                source_project_path=project_path,
                source_dataset_path=dataset_path,
                source_dataset_entry=entry,
                output_dir=root / "output",
                investigation_id="generic-investigation",
                trigger_reason="harmonic-cycle-preferred",
                primary_period_days=0.8001069097176371,
            )

        self.assertFalse(result["executable"])
        self.assertEqual(
            "frequency-window-outside-low-frequency-domain", result["reason"]
        )
        self.assertNotIn("frequencySearch", result)
        self.assertGreater(
            result["attemptedWindow"]["minimumFrequency"],
            result["attemptedWindow"]["maximumFrequency"],
        )


if __name__ == "__main__":
    unittest.main()
