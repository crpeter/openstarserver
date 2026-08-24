from __future__ import annotations

import copy
import math
import unittest

from workflows.tess.tess_target_residual_mechanism_predictive_validation import (
    PREDICTIVE_FOLDS, UNRESOLVED, construct_folds, fit_training_model,
    freeze_model_domain, validate_sector,
)


class TessTargetResidualMechanismPredictiveValidationTests(unittest.TestCase):
    def setUp(self):
        self.times = [index / 20 for index in range(400)]
        self.domain = freeze_model_domain(self.times, 1.0)

    def test_folds_are_deterministic_blocked_and_retain_every_segment(self):
        first = construct_folds(self.times, self.domain)
        self.assertEqual(first, construct_folds(self.times, self.domain))
        self.assertEqual(PREDICTIVE_FOLDS, len(first))
        self.assertTrue(all(len(fold["heldOutBlocks"]) == 5 for fold in first))
        self.assertTrue(all(block["count"] for fold in first for block in fold["heldOutBlocks"]))

    def test_domain_and_beat_grid_are_frozen_before_edge_holdout(self):
        folds = construct_folds(self.times, self.domain)
        self.assertIn(0, folds[0]["heldOutIndices"])
        for fold in folds:
            self.assertEqual(self.times[0], self.domain["fullDomainStart"])
            self.assertEqual(self.times[-1], self.domain["fullDomainEnd"])
            self.assertEqual(81, len(self.domain["beatDeltaGrid"]))

    def test_held_out_values_cannot_change_any_training_fit(self):
        values = [(1 + .5 * ((time - 10) / 10) ** 2) * math.sin(2*math.pi*time)
                  for time in self.times]
        fold = construct_folds(self.times, self.domain)[0]
        train = fold["trainingIndices"]
        altered = copy.copy(values)
        for index in fold["heldOutIndices"]: altered[index] += 1e9
        for model in ("CONSTANT_AMPLITUDE", "SMOOTH_AMPLITUDE_MODULATION",
                      "COHERENT_TWO_MODE_BEATING", "EPISODIC_ACTIVATION"):
            left = fit_training_model(model, [self.times[i] for i in train],
                                      [values[i] for i in train], 1.0, self.domain)
            right = fit_training_model(model, [self.times[i] for i in train],
                                       [altered[i] for i in train], 1.0, self.domain)
            self.assertEqual(left, right)

    def test_constant_predicts_best_but_does_not_claim_mechanism(self):
        result = validate_sector(self.times, [math.sin(2*math.pi*t) for t in self.times], 1.0,
            sector=1, dataset_id="one", timing_coordinate="SECTOR_LOCAL_WARPED_TIME",
            episodic_morphology=False)
        self.assertEqual("CONSTANT_AMPLITUDE", result["bestPredictiveModel"])
        self.assertEqual(UNRESOLVED, result["sectorClassification"])

    def test_smooth_beating_and_episodic_synthetic_signals(self):
        cases = (
            (lambda t: (1+.8*((t-10)/10)**2)*math.sin(2*math.pi*t), False,
             "SMOOTH_AMPLITUDE_MODULATION"),
            (lambda t: math.sin(2*math.pi*.96*t)+math.sin(2*math.pi*1.04*t), False,
             "COHERENT_TWO_MODE_BEATING"),
            (lambda t: (.05 if 8 <= t < 12 else 1)*math.sin(2*math.pi*t), True,
             "EPISODIC_ACTIVATION"),
        )
        for signal, morphology, expected in cases:
            with self.subTest(model=expected):
                result = validate_sector(self.times, [signal(t) for t in self.times], 1.0,
                    sector=1, dataset_id="one", timing_coordinate="SECTOR_LOCAL_WARPED_TIME",
                    episodic_morphology=morphology)
                self.assertEqual(expected, result["bestPredictiveModel"])
                self.assertNotEqual(UNRESOLVED, result["sectorClassification"])

    def test_episodic_predictive_win_obeys_frozen_morphology_veto(self):
        values = [(.05 if 8 <= t < 12 else 1)*math.sin(2*math.pi*t) for t in self.times]
        result = validate_sector(self.times, values, 1.0, sector=1, dataset_id="one",
            timing_coordinate="SECTOR_LOCAL_WARPED_TIME", episodic_morphology=False)
        self.assertEqual("EPISODIC_ACTIVATION", result["bestPredictiveModel"])
        self.assertTrue(result["morphologyGateBlockedPromotion"])
        self.assertEqual(UNRESOLVED, result["sectorClassification"])


if __name__ == "__main__":
    unittest.main()
