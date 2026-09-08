# Record a reviewed common-support localization

This offline continuation depends on PR #189. It accepts the finalized v1
conflicting-localization boundary and the exact v2 JSON file reviewed by the
operator. The v2 artifact must resolve to the target under the existing rules,
and every originally measured sector must remain usable and target-consistent.
Previously rejected sectors are preserved. Other outcomes require their own
review and are outside this continuation's scope.

The CLI requires `--reviewed-sha256`: use the digest of the file actually reviewed,
not a newly calculated digest used to bypass a mismatch. The digest binds the
operator's execution decision to exact bytes. There is no target-specific digest
or catalog answer key embedded in the implementation.

```bash
python -m unittest -v \
  test_tess_eclipse_common_support_continuation \
  test_tess_eclipse_common_support \
  test_tess_eclipse_event_localization

python run_openstar_tess_eclipse_common_support_continuation.py \
  --state-dir /durable/path/state \
  --investigation-id your-investigation-id \
  --reanalysis-file /durable/path/eclipse-common-support-v2.json \
  --reviewed-sha256 THE_EXACT_REVIEWED_FILE_SHA256
```

The default mode validates only and prints `VALIDATED_NO_CHANGES`, proposed stage
IDs, and output paths. Add `--execute` to record the result, yielding `RECORDED`.
Use a dedicated checkout and an idle investigation. No coordinator, worker,
archive request, catalog query, period search, or pixel reacquisition is involved.

Verification checks the current terminal ledgers, artifact hashes and original
localization boundary; reviewed-file digest; exact parent hashes and frozen event
definitions; method contract and source hashes; complete sector accounting;
saved difference/SNR maps against v1; positive-flux image moments; recorded event
jackknife aggregates; catalog distance identities and margins; and the reused
cross-sector summary. Source paths can move between checkouts, but their
repository-relative identity and exact content hashes must match.

This is not independent reproduction from raw cadences: the v2 JSON contains no
cadence cube or complete WCS. Event omission measurements, validity-mask ancestry,
catalog projections and catalog-overlap geometry remain bound to the reviewed
artifact and its producer provenance. The offline checks do not establish
absence of crowding, saturation, registration, clipping or background biases.

Execution appends two immutable terminal stages:

- `openstar.tess.eclipse-common-support.accept`: original reviewed v2 result and
  exact copies of the reviewed file and parent investigation snapshot.
- `openstar.tess.eclipse-common-support.finalize`: new versioned conclusion and
  report, stopping with `SOURCE_ATTRIBUTION_REVIEW` as the recommended next test.

Outputs are under the investigation's
`artifacts/eclipse-common-support-continuation-v1/` directory. The report is
`report-eclipse-common-support-v2.md`, and the conclusion is
`conclusion-eclipse-common-support-v2.json`. The original v1 localization,
conclusion, report, all existing terminal stages and metadata remain unchanged.
The current `investigation.json` snapshot gains the two completed stages. The
claim stays `CANDIDATE_PERIOD`; companion nature, physical mechanism and discovery
claims are not promoted. The new handlers have distinct identities and do not
relabel the result as v1. No subsequent science stage is scheduled; a future
continuation must explicitly consume this new terminal boundary.

The snapshot is committed last, after publishing new artifacts and ledgers.
Ordinary publication failures roll back this invocation's new files when the
snapshot has not committed. Directory locking serializes this CLI on macOS and
Linux; other writers do not share that lock, so keep the investigation idle.
Detected state or source changes refuse publication. A process kill before the
snapshot commit can leave an unreferenced output package or stage ledger; the
next run refuses collisions for inspection and never overwrites them.

Repeating the same successful execution verifies the recorded ledgers and
artifacts, returns `ALREADY_RECORDED`, and adds no duplicate stages. A different
reviewed artifact cannot replace the recorded continuation.

Tests, builds and science execution are left to the repository owner.
