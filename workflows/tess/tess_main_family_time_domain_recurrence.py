"""Independent time-domain test of an unresolved TESS photometric family.

This server-side module intentionally consumes flux (never periodogram power).
All thresholds are preregistered in :data:`METHOD`; callers persist that mapping.
"""
from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np

ROTATION_MULTICYCLE = "ROTATION_MULTICYCLE_RECURRENCE_SUPPORTED"
INDEPENDENT_LONG = "INDEPENDENT_LONGER_PERIOD_RECURRENCE_SUPPORTED"
NOT_REPLICATED = "FREQUENCY_FAMILY_NOT_TIME_DOMAIN_REPLICATED"
UNRESOLVED = "MAIN_FAMILY_TIME_DOMAIN_RELATION_UNRESOLVED"

METHOD = {
    "lagSearchDays": [1.0, 18.0],
    "lagGridStepDays": 0.05,
    "minimumPairSupport": 40,
    "minimumPeakCorrelation": 0.20,
    "minimumPeakProminence": 0.08,
    "jackknifeBlocks": 6,
    "minimumJackknifeDetections": 4,
    "phaseBins": 16,
    "minimumPopulatedPhaseBinFraction": 0.75,
    "minimumCyclePairsPerSeparation": 3,
    "replicatedSectorCount": 2,
    "interpretation": {
        "minimumPeakCorrelation": "positive temporal recurrence large enough to be scientifically useful",
        "minimumPeakProminence": "peak rise above its local ACF shoulders",
        "minimumPairSupport": "prevents sparse-lag correlations from masquerading as recurrence",
        "replicatedSectorCount": "independent-sector replication requirement",
    },
}


def _corr(x, y):
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _acf_at(time, flux, lag, half_width):
    """Gap-aware slot correlation: nearest observations within half a grid cell."""
    target = time + lag
    j = np.searchsorted(time, target)
    choices = np.clip(np.column_stack((j - 1, j)), 0, len(time) - 1)
    distances = np.abs(time[choices] - target[:, None])
    pick = choices[np.arange(len(time)), np.argmin(distances, axis=1)]
    mask = (np.min(distances, axis=1) <= half_width) & (pick > np.arange(len(time)))
    return _corr(flux[mask], flux[pick[mask]]), int(mask.sum())


def gap_aware_acf(time: Iterable[float], flux: Iterable[float], method=METHOD):
    time, flux = np.asarray(time, float), np.asarray(flux, float)
    good = np.isfinite(time) & np.isfinite(flux)
    time, flux = time[good], flux[good]
    order = np.argsort(time); time, flux = time[order], flux[order]
    flux = flux - np.median(flux)
    step = method["lagGridStepDays"]
    lags = np.arange(method["lagSearchDays"][0], method["lagSearchDays"][1] + step/2, step)
    values, support = [], []
    cadence = np.median(np.diff(time)) if len(time) > 1 else step
    for lag in lags:
        value, count = _acf_at(time, flux, lag, max(step / 2, cadence * 0.6))
        values.append(np.nan if value is None else value); support.append(count)
    return lags, np.asarray(values), np.asarray(support), time, flux


def _peaks(lags, acf, support, method=METHOD):
    peaks = []
    for i in range(1, len(lags)-1):
        if not (np.isfinite(acf[i]) and acf[i] >= acf[i-1] and acf[i] > acf[i+1]): continue
        shoulder = max(np.nanmin(acf[max(0,i-5):i+1]), np.nanmin(acf[i:min(len(acf),i+6)]))
        prominence = float(acf[i] - shoulder)
        if (acf[i] < method["minimumPeakCorrelation"] or prominence < method["minimumPeakProminence"]
                or support[i] < method["minimumPairSupport"]): continue
        half = acf[i] - prominence/2
        left = i
        while left and acf[left] >= half: left -= 1
        right = i
        while right < len(acf)-1 and acf[right] >= half: right += 1
        peaks.append({"lagDays": float(lags[i]), "peakCorrelation": float(acf[i]),
            "localPeakWidthDays": float(lags[right]-lags[left]), "overlapSampleSupport": int(support[i]),
            "prominence": prominence, "detectionCriteria": {k: method[k] for k in
            ("minimumPeakCorrelation","minimumPeakProminence","minimumPairSupport")}})
    return peaks


def _nearest(peaks, target, window):
    eligible = [p for p in peaks if abs(p["lagDays"]-target) <= window]
    return max(eligible, key=lambda p:p["peakCorrelation"], default=None)


def analyze_sector(time, flux, *, sector_id, rotation_period_days,
                   possible_double_days=None, method=METHOD):
    lags, acf, support, time, flux = gap_aware_acf(time, flux, method)
    peaks = _peaks(lags, acf, support, method)
    rotation = _nearest(peaks, rotation_period_days, max(.5, 2*method["lagGridStepDays"]))
    # Delete contiguous time blocks, recompute the continuous search, and retain
    # the peak nearest the full-sample peak (not a preselected physical lag).
    samples = []
    if rotation and len(time):
        edges = np.linspace(time.min(), time.max(), method["jackknifeBlocks"]+1)
        for block in range(method["jackknifeBlocks"]):
            keep = ~((time >= edges[block]) & (time <= edges[block+1]))
            jl, ja, js, _, _ = gap_aware_acf(time[keep], flux[keep], method)
            candidate = _nearest(_peaks(jl, ja, js, method), rotation["lagDays"],
                                 max(rotation["localPeakWidthDays"], .5))
            samples.append(None if candidate is None else candidate["lagDays"])
    detected = np.asarray([x for x in samples if x is not None], float)
    uncertainty = {"method":"deterministic-delete-one-contiguous-time-block-jackknife",
        "numberOfBlocks":method["jackknifeBlocks"], "perResamplePeakLocationsDays":samples,
        "medianLagDays":float(np.median(detected)) if len(detected) else None,
        "meanLagDays":float(np.mean(detected)) if len(detected) else None,
        "intervalDays":[float(np.min(detected)),float(np.max(detected))] if len(detected) else None}
    if rotation: rotation["uncertaintyEstimate"] = uncertainty
    # Cycle profiles and correlations.
    cycle = np.floor((time-time.min())/rotation_period_days).astype(int)
    phase = ((time-time.min())/rotation_period_days) % 1
    profiles = {}
    for c in np.unique(cycle):
        profile=[]
        for b in range(method["phaseBins"]):
            x=flux[(cycle==c)&(phase>=b/method["phaseBins"])&(phase<(b+1)/method["phaseBins"])]
            profile.append(float(np.median(x)) if len(x) else np.nan)
        if np.isfinite(profile).mean() >= method["minimumPopulatedPhaseBinFraction"]: profiles[int(c)]=np.asarray(profile)
    pairs={str(k):[] for k in (1,2,4)}
    for separation in (1,2,4):
        for c,p in profiles.items():
            if c+separation not in profiles: continue
            q=profiles[c+separation]; valid=np.isfinite(p)&np.isfinite(q)
            value=_corr(p[valid],q[valid])
            if value is not None: pairs[str(separation)].append({"firstCycle":c,"secondCycle":c+separation,"correlation":value,"commonPhaseBins":int(valid.sum())})
    summaries={k:{"pairCount":len(v),"medianCorrelation":float(np.median([x["correlation"] for x in v])) if v else None,
        "deleteOnePairJackknifeMedians":[float(np.median([y["correlation"] for j,y in enumerate(v) if j!=i])) for i in range(len(v))] if len(v)>1 else []}
        for k,v in pairs.items()}
    baseline=float(time.max()-time.min()) if len(time) else 0
    return {"sectorID":sector_id,"timeBaselineDays":baseline,"acfPeaks":peaks,
        "rotationRecurrencePeak":rotation,"rotationJackknife":uncertainty,
        "canConstrainPossibleDouble":(possible_double_days is not None and
            baseline >= 2*float(possible_double_days)),
        "cyclePairMeasurements":pairs,"cycleSeparationSummaries":summaries,
        "cyclesAccepted":sorted(profiles),"coverageCriteria":{k:method[k] for k in
        ("phaseBins","minimumPopulatedPhaseBinFraction","minimumCyclePairsPerSeparation")}}


def combine_sector_results(results, *, rotation_period_days, family_period_days,
                           possible_double_days, method=METHOD):
    related=[]; independent=[]; long_detected=[]; morphology=[]
    for result in results:
        rotation=result.get("rotationRecurrencePeak")
        if not rotation: continue
        interval=(rotation.get("uncertaintyEstimate") or {}).get("intervalDays")
        sigma=max((interval[1]-interval[0])/2 if interval else 0, rotation["localPeakWidthDays"]/2,
                  method["lagGridStepDays"])
        long_peak=_nearest(result["acfPeaks"], family_period_days,
                           max(sigma, method["lagGridStepDays"]*2))
        if long_peak:
            long_detected.append(result["sectorID"])
            distance=min(abs(long_peak["lagDays"]-2*rotation["lagDays"]),abs(long_peak["lagDays"]-4*rotation["lagDays"]))
            (related if distance <= sigma + long_peak["localPeakWidthDays"]/2 else independent).append(result["sectorID"])
        s=result["cycleSeparationSummaries"]
        valid=[(int(k),v["medianCorrelation"]) for k,v in s.items() if v["pairCount"]>=method["minimumCyclePairsPerSeparation"] and v["medianCorrelation"] is not None]
        if valid:
            best=max(valid,key=lambda x:x[1])
            if best[0] in (2,4): morphology.append(result["sectorID"])
    n=method["replicatedSectorCount"]
    if len(related)>=n and len(morphology)>=n: classification=ROTATION_MULTICYCLE
    elif len(independent)>=n: classification=INDEPENDENT_LONG
    elif len(results)>=n and len(long_detected)==0: classification=NOT_REPLICATED
    else: classification=UNRESOLVED
    relationship=classification in (ROTATION_MULTICYCLE,INDEPENDENT_LONG)
    exact=False
    next_test={ROTATION_MULTICYCLE:"LONG_BASELINE_ACTIVE_REGION_RECURRENCE_CONFIRMATION",
        INDEPENDENT_LONG:"INDEPENDENT_LONG_PERIOD_ASTROPHYSICAL_INTERPRETATION",
        NOT_REPLICATED:"MAIN_FAMILY_FREQUENCY_DOMAIN_REASSESSMENT",
        UNRESOLVED:"LONG_BASELINE_TIME_DOMAIN_RECURRENCE_DATA"}[classification]
    gates={"authoritativeRotationMechanismResolved":True,"authoritativeRotationPeriodAvailable":rotation_period_days>0,
        "frozenMainFamilyAvailable":family_period_days>0,"sufficientTimeDomainCoverage":len(results)>=n,
        "acfRotationRecurrenceDetected":sum(bool(r.get("rotationRecurrencePeak")) for r in results)>=n,
        "replicatedLongLagRecurrence":len(long_detected)>=n,"rotationMultipleConsistency":len(related)>=n,
        "cycleMorphologySupportsSameRelation":len(morphology)>=n,"independentLongPeriodEvidence":len(independent)>=n,
        "noStrongerContradiction":not (related and independent)}
    return {"classification":classification,"mainFamilyRelationshipToRotationResolved":relationship,
        "mainFamilyRelationshipClassification":classification,"physicalCycleResolved":exact,
        "exactPhysicalCycleResolved":exact,"recommendedNextTest":next_test,"decisionGates":gates,
        "relatedSectorIDs":related,"independentSectorIDs":independent,"longLagSectorIDs":long_detected,
        "morphologySupportingSectorIDs":morphology}


def analyze_time_domain_recurrence(sectors, *, rotation_period_days, rotation_classification,
                                   main_photometric_family, method=METHOD):
    double=float(main_photometric_family["possibleDoubleCycleDays"])
    results=[analyze_sector(s["time"],s["flux"],sector_id=s["sectorID"],
        rotation_period_days=rotation_period_days,possible_double_days=double,
        method=method) for s in sectors]
    family=float(main_photometric_family["representativeRawPeriodDays"])
    combined=combine_sector_results(results,rotation_period_days=rotation_period_days,
        family_period_days=family,possible_double_days=double,method=method)
    return {"schemaVersion":"main-family-time-domain-recurrence-v1",
        "authoritativeRotationPeriodDays":rotation_period_days,
        "authoritativeRotationClassification":rotation_classification,
        "mainPhotometricFamily":dict(main_photometric_family),"sectorsEvaluated":[s["sectorID"] for s in sectors],
        "acfMethod":{"name":"gap-aware-normalized-slot-autocorrelation","parameters":method},
        "acfSectorResults":[{k:v for k,v in r.items() if k not in ("cyclePairMeasurements","cycleSeparationSummaries","cyclesAccepted","coverageCriteria")} for r in results],
        "cycleRecurrenceMethod":{"name":"rotation-clock-phase-binned-profile-correlation","parameters":method},
        "cycleRecurrenceSectorResults":[{"sectorID":r["sectorID"],"cyclePairMeasurements":r["cyclePairMeasurements"],"cycleSeparationSummaries":r["cycleSeparationSummaries"],"cyclesAccepted":r["cyclesAccepted"],"coverageCriteria":r["coverageCriteria"]} for r in results],
        "combinedEvidence":combined,"decisionGates":combined["decisionGates"],
        "classification":combined["classification"],"mainFamilyRelationshipToRotationResolved":combined["mainFamilyRelationshipToRotationResolved"],
        "mainFamilyRelationshipClassification":combined["mainFamilyRelationshipClassification"],"physicalCycleResolved":combined["physicalCycleResolved"],
        "exactPhysicalCycleResolved":combined["exactPhysicalCycleResolved"],"recommendedNextTest":combined["recommendedNextTest"],
        "interpretationNotes":["The solved rotation is immutable and was not re-estimated by Lomb-Scargle.","Relationship resolution is distinct from exact physical-cycle resolution."]}
