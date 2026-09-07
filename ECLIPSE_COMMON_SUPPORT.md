# Eclipse localization with common spatial support

The v1 eclipse estimator selects a 2.5-pixel neighborhood around the highest
signal-to-noise pixel. A high-SNR wing can therefore exclude the bright core of
the same event image. Multiplying by SNR also changes the effective spatial
response. The separately identified v2 reanalysis removes both operations from
centroid measurement. It is opt-in; the existing investigation path stays v1.

## Method

`openstar.tess-eclipse-common-support.v1` measures the positive-flux moments of
the out-of-event and out-minus-in images over **all valid pixels in the original
stamp**. Both measurements and every leave-one-event-out replicate use the same
validity mask, fixed from the original full selected cadence cube. There is no
target-centered crop, highest-SNR-centered crop, SNR pixel cutoff, or SNR centroid
weight. The signed total event loss and out-of-event reference flux must be
positive. Peak SNR is retained only for the existing detection-quality gate.

The original event period, epochs, durations, quality selection, cadence cap,
background correction, filling rule, event/control bins and opposite-conjunction
exclusion are preserved. The revised centroid is recomputed for every event
jackknife; the original uncertainty floor and quality threshold are retained.
Catalog matching and cross-sector decisions use the existing functions and
thresholds: the primary sector does not satisfy independent replication, and a
usable conflicting independent sector cannot be outvoted.

The out-of-event centroid is a **scene reference**, not a calibrated target
position. Its offset is diagnostic and is never subtracted before catalog
matching. Difference centroids remain in the original detector coordinates;
off-catalog sky comparisons require newly projected WCS positions. Missing WCS
never reuses a previous centroid's sky coordinates.

Positive clipping, crowding, saturation, background errors and finite stamps can
still bias these moments. A stable jackknife does not establish absence of those
systematics. This is not a calibrated PRF fit, a source exclusion based on catalog
magnitude, or evidence about companion nature. Review the separate result before
considering a future investigation continuation.

## Run locally

Use a dedicated checkout while other investigations run, so switching branches
in another terminal cannot remove the executing source files.

```bash
python -m unittest -v test_tess_eclipse_common_support test_tess_eclipse_event_localization

python run_openstar_tess_eclipse_common_support.py \
  --state-dir /durable/path/state \
  --investigation-id your-investigation-id \
  --output-file /durable/path/eclipse-common-support-v2.json
```

The default command validates only, returning `VALIDATED_NO_CHANGES`. It requires
the finalized v1 `CROSS_SECTOR_SOURCE_DISAGREEMENT_OR_BLEND` boundary, verified
terminal ledgers and artifacts, exact binary/catalog ancestry, the production
stage's catalog/pixel hash metadata, and complete original sector accounting.
It neither acquires pixels nor writes output.

Add `--execute` after validation to reacquire the original measured sectors and
write a separate result with identity
`openstar.tess-eclipse-event-source-localization.v2` and status
`REANALYSIS_REQUIRES_REVIEW`. Every original pixel hash and v1 difference/SNR image
and centroid diagnostic must reproduce before the new estimator runs. Previously
rejected sectors remain rejected; every previously measured sector remains
visible even if the new estimator cannot measure it. New acquisition failures
abort rather than silently discarding a conflicting sector.

No coordinator or worker is involved. The command does not modify investigation
state, old results, reports, claims, or acceptance thresholds. Its separate
`sourceAttributionResolved` field describes only the v2 result under the existing
policy, not an update to the investigation. The output includes old/new per-sector
positions, paired images, masks, new jackknife results, parent hashes, a method
contract and source hashes. Sources are checked before acquisition and again
before publication; source changes or state changes refuse output. Output must
be outside state and must not already exist. Publication is atomic and refuses
overwrites.

This revision is independent of the diagnostic-only PR #188 and does not change
its audit or the shared non-eclipse difference-image estimator. Tests and archive
or science runs are left to the repository owner.
