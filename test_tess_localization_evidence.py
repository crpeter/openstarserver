import tempfile
import unittest
import sys
import types
from dataclasses import replace
from pathlib import Path
from unittest import mock

try:
    import numpy  # noqa: F401
except ImportError:
    sys.modules["numpy"] = types.ModuleType("numpy")

from openstar_investigation import InvestigationStage, InvestigationStore
from openstar_lifecycle import InvestigationLifecycleDriver
from openstar_targets import InvestigationTarget
from openstar_workflow import RetryableExecutionError, StageRequest
from workflows.tess.tess_investigation import build_engine
from workflows.tess.tess_localization_evidence import (
    frozen_residual_localization_family,
)
from workflows.tess.tess_residual_localization import build_residual_mode_pixel_project
from workflows.tess.tess_residual_localization_review import (
    build_residual_mode_localization_review_project,
)
from workflows.tess.tess_sector_archive import TessArchiveTransientError


class FrozenResidualLocalizationFamilyTests(unittest.TestCase):
    def _evidence(self, *, morphology_period=10.510316195053623):
        morphology = {
            "physicalCycleResolved": True,
            "resolvedPhysicalPeriodDays": morphology_period,
        }
        time_frequency_prepare = {"absoluteTimeReferenceDays": 2500.0}
        time_frequency = {
            "residualEvolution": {"classification": "STABLE_RESIDUAL_MODE"}
        }
        mode = {
            "classification": "INDEPENDENT_STABLE_MODE",
            "independentModeEvidenceSurvived": True,
            "physicalMechanismResolved": False,
            "establishedPeriodFamily": {
                "referencePeriodDays": 10.510316195053623,
                "modeledHarmonicOrders": [1, 2, 3],
            },
            "modeCandidate": {
                "frequencyCyclesPerDay": 0.27101611598985065,
                "periodDays": 1.0 / 0.27101611598985065,
                "supportingSectors": [28, 68, 92, 95],
            },
        }
        return morphology, time_frequency_prepare, time_frequency, mode

    def test_real_resolved_family_adapts_and_preserves_morphology_period(self):
        morphology, tf_prepare, tf_summary, mode = self._evidence()

        adapted = frozen_residual_localization_family(
            morphology, None, tf_prepare, tf_summary, mode
        )

        self.assertIsNotNone(adapted)
        self.assertEqual(10.510316195053623, adapted[0])
        self.assertEqual((1, 2, 3), adapted[1])
        self.assertEqual("MORPHOLOGY_RESOLVED_PHYSICAL_PERIOD", adapted[3])

    def test_resolved_morphology_and_mode_family_mismatch_fails_closed(self):
        morphology, tf_prepare, tf_summary, mode = self._evidence(
            morphology_period=11.510316195053623
        )

        self.assertIsNone(frozen_residual_localization_family(
            morphology, None, tf_prepare, tf_summary, mode
        ))

    def test_actual_review_prepare_rejects_mismatched_resolved_family(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        store = InvestigationStore(Path(temporary.name) / "investigations")
        investigation = store.create("inconsistent-review", "workflow", "1")
        morphology, tf_prepare, tf_summary, mode = self._evidence(
            morphology_period=11.510316195053623
        )
        stages = (
            InvestigationStage("001-prepare-target", "openstar.tess.prepare-target",
                               "COMPLETE", None, {}, result={}),
            InvestigationStage("002-identity", "openstar.tess.catalog-identity",
                               "COMPLETE", "001-prepare-target", {}, result={}),
            InvestigationStage("003-independent", "openstar.tess.independent.prepare",
                               "COMPLETE", "002-identity", {}, result={}),
            InvestigationStage("010-morphology", "openstar.tess.morphology.analyze",
                               "COMPLETE", "003-independent", {}, result=morphology),
            InvestigationStage("012-time-frequency-prepare",
                               "openstar.tess.time-frequency.prepare", "COMPLETE",
                               "010-morphology", {}, result=tf_prepare),
            InvestigationStage("013-time-frequency-summary",
                               "openstar.tess.time-frequency.summarize", "COMPLETE",
                               "012-time-frequency-prepare", {}, result=tf_summary),
            InvestigationStage("018-mode-identification",
                               "openstar.tess.mode-identification.analyze", "COMPLETE",
                               "013-time-frequency-summary", {}, result=mode),
            InvestigationStage("021-interpret-residual-mode-localization",
                               "openstar.tess.residual-mode-localization.interpret",
                               "COMPLETE", "018-mode-identification", {}, result={
                                   "recommendedNextTest":
                                   "RESIDUAL_MODE_SOURCE_LOCALIZATION_REVIEW"
                               }),
        )
        investigation = replace(investigation, status="RUNNING", stages=stages)
        store.save(investigation)
        request = StageRequest(
            "022-prepare-residual-mode-localization-review",
            "openstar.tess.residual-mode-localization-review.prepare", {},
            "021-interpret-residual-mode-localization",
        )
        engine = build_engine(store, mock.Mock(), poll_interval=0.0, timeout=1.0)

        with mock.patch(
            "workflows.tess.tess_investigation."
            "build_residual_mode_localization_review_project"
        ) as build_project:
            with self.assertRaisesRegex(RuntimeError, "completed v20.9 nonstationary"):
                engine.run_stage(
                    investigation, request, software_id="test", software_version="current"
                )
        build_project.assert_not_called()

    def _archive_failure_investigation(self, *, review):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        store = InvestigationStore(Path(temporary.name) / "investigations")
        investigation = store.create("archive-timeout", "workflow", "1")
        source_project = Path(temporary.name) / "source.json"
        source_project.write_text("{}", encoding="utf-8")
        morphology, tf_prepare, tf_summary, mode = self._evidence()
        mode["recommendedNextTest"] = "RESIDUAL_MODE_PIXEL_LOCALIZATION"
        stages = [
            InvestigationStage(
                "001-prepare-target", "openstar.tess.prepare-target", "COMPLETE",
                None, {}, result={"sourceProjectPath": str(source_project),
                                  "sourceDatasetEntry": {}, "ticID": 42, "sector": 28},
            ),
            InvestigationStage("002-identity", "openstar.tess.catalog-identity",
                               "COMPLETE", "001-prepare-target", {}, result={
                                   "tic": {"metadata": {"raDeg": 1.0, "decDeg": 2.0}}
                               }),
            InvestigationStage("003-independent", "openstar.tess.independent.prepare",
                               "COMPLETE", "002-identity", {}, result={}),
            InvestigationStage("010-morphology", "openstar.tess.morphology.analyze",
                               "COMPLETE", "003-independent", {}, result=morphology),
            InvestigationStage("012-time-frequency-prepare",
                               "openstar.tess.time-frequency.prepare", "COMPLETE",
                               "010-morphology", {}, result=tf_prepare),
            InvestigationStage("013-time-frequency-summary",
                               "openstar.tess.time-frequency.summarize", "COMPLETE",
                               "012-time-frequency-prepare", {}, result=tf_summary),
            InvestigationStage("018-mode-identification",
                               "openstar.tess.mode-identification.analyze", "COMPLETE",
                               "013-time-frequency-summary", {}, result=mode),
        ]
        if review:
            stages.append(InvestigationStage(
                "021-interpret-residual-mode-localization",
                "openstar.tess.residual-mode-localization.interpret", "COMPLETE",
                "018-mode-identification", {}, result={
                    "recommendedNextTest": "RESIDUAL_MODE_SOURCE_LOCALIZATION_REVIEW"
                },
            ))
        investigation = replace(investigation, status="RUNNING", stages=tuple(stages))
        store.save(investigation)
        label = ("prepare-residual-mode-localization-review" if review
                 else "prepare-residual-mode-localization")
        request = StageRequest(
            f"022-{label}", f"openstar.tess.{label.replace('prepare-', '')}.prepare",
            {}, stages[-1].id,
        )
        return store, investigation, request

    def _assert_actual_prepare_timeout_is_retryable(self, *, review):
        store, investigation, request = self._archive_failure_investigation(review=review)
        engine = build_engine(store, mock.Mock(), poll_interval=0.0, timeout=1.0)
        download_module = ("tess_residual_localization_review" if review
                           else "tess_residual_localization")
        class _Degree:
            def __rmul__(self, value):
                return value
        coordinates = types.ModuleType("astropy.coordinates")
        coordinates.SkyCoord = lambda *args, **kwargs: object()
        units = types.ModuleType("astropy.units")
        units.deg = _Degree()
        astropy = types.ModuleType("astropy")
        astropy.coordinates, astropy.units = coordinates, units
        with mock.patch.dict(sys.modules, {
            "astropy": astropy, "astropy.coordinates": coordinates,
            "astropy.units": units,
        }), mock.patch(
            f"workflows.tess.{download_module}._download_tpf",
            side_effect=TessArchiveTransientError("MAST read timed out"),
        ) as download_tpf, mock.patch(
            "workflows.tess.tess_residual_localization._target_coordinate",
            return_value=object(),
        ):
            prepared = investigation.stages[0].result
            identity = investigation.stages[1].result
            nonstationary = {
                "preferredFrequencyAtReference": 0.27101611598985065,
                "fractionalFrequencyDriftPerDay": 0.0,
                "timeReferenceDays": 0.0,
                "preferredModel": {"signalSectors": [1]},
            }
            kwargs = dict(
                source_project_path=prepared["sourceProjectPath"],
                source_dataset_entry=prepared["sourceDatasetEntry"], tic_id=42,
                identity=identity, primary_sector=1, independent_spec={},
                physical_period_days=10.510316195053623,
                nonstationary_summary=nonstationary,
                output_dir=store.directory_for(investigation.id) / "direct-builder",
                investigation_id=investigation.id,
            )
            if review:
                builder = build_residual_mode_localization_review_project
                kwargs["residual_localization_summary"] = {
                    "recommendedNextTest":
                    "RESIDUAL_MODE_SOURCE_LOCALIZATION_REVIEW",
                    "sectorResults": [],
                }
            else:
                builder = build_residual_mode_pixel_project
            with self.assertRaises(TessArchiveTransientError):
                builder(**kwargs)
            if not review:
                download_tpf.side_effect = RuntimeError("genuine unusable pixels")
                with self.assertRaisesRegex(
                    RuntimeError, "could not prepare any residual-mode pixel datasets"
                ):
                    builder(**kwargs)
                download_tpf.side_effect = TessArchiveTransientError(
                    "MAST read timed out"
                )
            with self.assertRaises(RetryableExecutionError):
                engine.run_stage(
                    investigation, request, software_id="test", software_version="1"
                )

        failed = store.load(investigation.id)
        self.assertEqual("FAILED", failed.status)
        self.assertEqual("FAILED", failed.stages[-1].status)
        self.assertEqual("TRANSIENT_INFRASTRUCTURE",
                         failed.stages[-1].failure_classification)
        historical_bytes = store.stage_path_for(failed.id, request.id).read_bytes()

        target = InvestigationTarget("42", failed.id, "workflow", "1")
        driver = InvestigationLifecycleDriver(
            store, mock.Mock(), {}, software_id="test", software_version="1"
        )
        first = driver.prepare(target).investigation
        selected = first.metadata["controlState"]["selectedExperiment"]
        self.assertNotEqual(request.id, selected["id"])
        self.assertEqual(request.id, selected["triggered_by_stage_id"])
        second = driver.prepare(target).investigation
        self.assertEqual(selected, second.metadata["controlState"]["selectedExperiment"])
        self.assertEqual(historical_bytes,
                         store.stage_path_for(failed.id, request.id).read_bytes())
        self.assertEqual(failed.stages, second.stages)

    def test_actual_residual_prepare_timeout_is_retryable_and_restarted_once(self):
        self._assert_actual_prepare_timeout_is_retryable(review=False)

    def test_actual_review_prepare_timeout_is_retryable_and_restarted_once(self):
        self._assert_actual_prepare_timeout_is_retryable(review=True)


if __name__ == "__main__":
    unittest.main()
