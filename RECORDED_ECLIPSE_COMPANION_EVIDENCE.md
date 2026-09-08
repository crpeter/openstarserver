# Continue companion evidence from recorded localization v2

This continuation builds on PR #191. It consumes the recorded common-support v2
accept/finalize stages and resumes the existing source-attribution review and
companion-evidence pipeline. No original localization or recorded review file is
rewritten, and v2 is never relabeled as v1.

The v1 review API keeps its existing default gate. The workflow enables v2 only
after checking the recorded stages, immutable ledgers and artifacts, preserved
parent snapshot, exact reviewed bytes, source hashes, frozen sector evidence and
the v2 policy summary. The review method remains v1 because its scientific rules
are unchanged; a v2 review explicitly carries `sourceLocalizationVersion` and
the SHA-256 of the actual v2 localization. Synthesis recomputes that exact review
and preserves the existing external evidence and mass-regime requirements.

The stage order remains:

1. Review source attribution using the recorded localization.
2. Freeze full-precision photometry and audit eclipse-depth attenuation.
3. Fit the existing joint event/phase model when the depth audit requires it.
4. Freeze the external companion response, then interpret it.
5. Synthesize resolved companion evidence when the existing gates permit it,
   and finalize the investigation.

The existing depth-audit policy can proceed to external evidence with unavailable
precision photometry; unresolved depth/model diagnostics remain explicit. This
PR does not relax those rules or claim that the model must resolve. It does not
change event timing, numerical fitting, catalog matching, mass thresholds, claim
promotion or discovery policy. External published-object knowledge enters only
at its existing stage after the software-blind photometric checks.

## Local commands

Use a dedicated checkout and an idle investigation. No coordinator or worker is
required by this path. Default validation makes no archive requests or writes.

```bash
python -m unittest -v \
  test_tess_recorded_eclipse_evidence \
  test_tess_eclipse_common_support_continuation \
  test_tess_external_companion_evidence \
  test_tess_companion_evidence_synthesis \
  test_tess_event_depth_accuracy \
  test_tess_joint_event_phase_model

python run_openstar_tess_recorded_eclipse_evidence.py \
  --state-dir /durable/path/state \
  --investigation-id your-investigation-id
```

Expect `VALIDATED_NO_CHANGES`, a deterministic source-review preview, and the next
stage. Add `--execute` to run the existing pipeline. This performs photometry
downloads, local numerical analysis, and the prescribed external query on the
owner's machine. It appends new stages using the shared workflow engine.

The final report and conclusion use a `-common-support-v2` filename suffix and
contain the recorded v2 localization. Original v1 and PR #191 reports remain
unchanged. The terminal output prints the new report and conclusion paths.

Re-running resumes a persisted next-stage request without rerunning completed
stages. After a transient photometry or external-archive acquisition failure,
add `--retry-failed` (with `--execute` to run it); the failed ledger is retained
and a new attempt is appended. Other failures and an interrupted RUNNING stage
require inspection rather than automatic retry. A completed finalizer returns
`ALREADY_COMPLETE` after verification, with no new stages or queries.

The directory lock serializes this continuation with PR #191's CLI on macOS and
Linux. Other workflow writers do not use that lock; keep the investigation idle.
The offline provenance checks have the same raw-cadence/WCS limitations documented
in PR #191. Tests, builds and science execution are left to the repository owner.
