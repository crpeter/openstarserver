"""Independent published companion evidence for a frozen eclipse localization.

This module deliberately does no photometric search or fitting.  It reviews the
already-frozen spatial attribution, freezes an exact NASA Exoplanet Archive TAP
response, and only then interprets that durable response.
"""
from __future__ import annotations

import hashlib
import json
import math
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Any, Callable

from openstar_investigation import sha256_json


LOCALIZATION_VERSION = "openstar.tess-eclipse-event-source-localization.v1"
REVIEW_VERSION = "openstar.tess-source-attribution-review.v1"
FREEZE_VERSION = "openstar.nasa-exoplanet-archive-companion-evidence-freeze.v1"
RESULT_VERSION = "openstar.external-companion-evidence.v1"
REVIEW_HANDLER_ID = "openstar.tess.eclipse-source-attribution.review"
FREEZE_HANDLER_ID = "openstar.tess.external-companion-evidence.freeze"
INTERPRET_HANDLER_ID = "openstar.tess.external-companion-evidence.interpret"
TAP_ENDPOINT = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
HTTP_TIMEOUT_SECONDS = 45
MIN_INDEPENDENT_SUPPORT = 3
# Preregistered absolute floor; published 1-sigma errors are added below.
PERIOD_MATCH_ABSOLUTE_TOLERANCE_DAYS = 0.01

FIELDS = ("pl_name", "hostname", "tic_id", "gaia_dr3_id", "default_flag", "soltype",
          "pl_controv_flag", "discoverymethod", "disc_refname", "pl_refname", "rv_flag",
          "tran_flag", "obm_flag", "pl_orbper", "pl_orbpererr1", "pl_orbpererr2",
          "pl_bmassj", "pl_bmassjerr1", "pl_bmassjerr2", "pl_bmassjlim", "pl_bmassprov",
          "pl_orbincl", "pl_orbinclerr1", "pl_orbinclerr2")


class ExternalEvidenceTransientError(RuntimeError):
    pass


def localization_gate(value: dict[str, Any]) -> bool:
    return (value.get("resultVersion") == LOCALIZATION_VERSION
            and value.get("sourceAttributionResolved") is True
            and value.get("classification") in {
                "TARGET_CONSISTENT_ECLIPSE_SOURCE",
                "OFF_TARGET_CATALOG_CANDIDATE_ECLIPSE_SOURCE",
                "CONSISTENTLY_OFF_TARGET_ECLIPSE_SOURCE"}
            and value.get("recommendedNextTest") == "SOURCE_ATTRIBUTION_REVIEW"
            and value.get("pixelDataChangedFrozenEventDefinition") is False
            and value.get("catalogAnswerKeyUsed") is False
            and value.get("physicalMechanismResolved") is False
            and value.get("companionNatureResolved") is False)


def _source_key(item: dict[str, Any]) -> str | None:
    if item.get("matchedCatalogHypothesis"):
        return str(item["matchedCatalogHypothesis"])
    if item.get("classification") == "OFF_CATALOG":
        return "OFF_CATALOG_SKY_CLUSTER"
    return None


def review_source_attribution(localization: dict[str, Any]) -> dict[str, Any]:
    if not localization_gate(localization):
        raise ValueError("exact localization-v1 source-attribution gate is not satisfied")
    attributed = localization.get("attributedCatalogHypothesis")
    support, conflicts, ambiguous = [], [], []
    for sector in localization.get("sectorResults") or []:
        if sector.get("role") != "INDEPENDENT":
            continue
        key = _source_key(sector) if sector.get("usable") is True else None
        if key is None:
            ambiguous.append(sector.get("sector"))
        elif key == attributed:
            support.append(sector.get("sector"))
        else:
            conflicts.append({"sector": sector.get("sector"), "source": key})
    catalog = localization.get("frozenCatalog") or {}
    hypotheses = catalog.get("catalogHypotheses") or []
    matches = [row for row in hypotheses if str(row.get("sourceID")) == str(attributed)]
    off_catalog = attributed == "OFF_CATALOG_SKY_CLUSTER"
    passed = len(support) >= MIN_INDEPENDENT_SUPPORT and not conflicts
    source = matches[0] if len(matches) == 1 else None
    if not passed:
        classification = "SOURCE_ATTRIBUTION_REVIEW_FAILED"
    elif off_catalog:
        classification = "UNCATALOGUED_SOURCE_REQUIRES_FOLLOWUP"
    elif source is None or source.get("ticID") in (None, ""):
        classification = "SOURCE_ATTRIBUTION_REVIEW_FAILED"
    elif localization["classification"] == "TARGET_CONSISTENT_ECLIPSE_SOURCE":
        classification = "TARGET_SOURCE_ATTRIBUTION_REVIEW_PASSED"
    else:
        classification = "OFF_TARGET_CATALOG_ATTRIBUTION_REVIEW_PASSED"
    return {"resultVersion": REVIEW_VERSION, "classification": classification,
            "sourceAttributionReviewPassed": classification.endswith("REVIEW_PASSED"),
            "supportingIndependentSectors": support, "supportingIndependentSectorCount": len(support),
            "ambiguousIndependentSectors": ambiguous, "conflictingIndependentSectors": conflicts,
            "primarySectorCanSatisfyReplication": False, "requiredIndependentSectorCount": MIN_INDEPENDENT_SUPPORT,
            "attributedSource": source, "attributedCatalogHypothesis": attributed,
            "sourceLocalizationSHA256": sha256_json(localization),
            "binaryConfirmationSHA256": localization.get("binaryConfirmationSHA256"),
            "frozenEphemeris": localization.get("frozenEphemeris"), "catalogAnswerKeyUsed": False,
            "physicalMechanismResolved": False, "companionNatureResolved": False,
            "recommendedNextTest": ("EXTERNAL_COMPANION_EVIDENCE_FREEZE" if classification.endswith("REVIEW_PASSED")
                                    else "CATALOG_IDENTITY_FOLLOWUP" if off_catalog
                                    else "ADDITIONAL_SPATIAL_EVIDENCE")}


def _canonical_tic(value: Any) -> str:
    text = str(value).strip()
    if text.upper().startswith("TIC "):
        number = text[4:].strip()
    elif text.isdigit():
        number = text
    else:
        raise ValueError("attributed source has malformed TIC identifier")
    if not number.isdigit() or int(number) <= 0:
        raise ValueError("attributed source has malformed TIC identifier")
    return f"TIC {int(number)}"


def build_tap_query(tic_id: str) -> tuple[str, str]:
    if "'" in tic_id:
        raise ValueError("unsafe TIC identifier")
    adql = f"select {','.join(FIELDS)} from ps where default_flag = 1 and tic_id = '{tic_id}'"
    encoded = urllib.parse.urlencode({"query": adql, "format": "json"})
    return adql, encoded


def acquire_external_evidence(review: dict[str, Any], *, opener: Callable[..., Any] = urllib.request.urlopen,
                              retrieved_at: str | None = None) -> dict[str, Any]:
    if review.get("resultVersion") != REVIEW_VERSION or review.get("sourceAttributionReviewPassed") is not True:
        raise ValueError("passed source-attribution review is required")
    source = review.get("attributedSource") or {}
    tic = _canonical_tic(source.get("ticID"))
    gaia = source.get("gaiaDR3SourceID")
    _, encoded = build_tap_query(tic)
    request = urllib.request.Request(f"{TAP_ENDPOINT}?{encoded}", headers={"User-Agent": "OpenStarServer/1"})
    try:
        with opener(request, timeout=HTTP_TIMEOUT_SECONDS) as response:
            status_value = getattr(response, "status", None)
            status = int(status_value if status_value is not None else response.getcode())
            content_type = response.headers.get("Content-Type", "")
            raw = response.read()
    except urllib.error.HTTPError as error:
        if error.code == 429 or 500 <= error.code < 600:
            raise ExternalEvidenceTransientError(f"NASA Exoplanet Archive HTTP {error.code}") from error
        raise ValueError(f"NASA Exoplanet Archive HTTP {error.code}") from error
    except (TimeoutError, ConnectionError, urllib.error.URLError) as error:
        raise ExternalEvidenceTransientError(f"NASA Exoplanet Archive unavailable: {error}") from error
    if status == 429 or status >= 500:
        raise ExternalEvidenceTransientError(f"NASA Exoplanet Archive HTTP {status}")
    if status != 200:
        raise ValueError(f"unexpected NASA Exoplanet Archive HTTP {status}")
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("malformed NASA Exoplanet Archive response") from error
    if not isinstance(parsed, list) or any(not isinstance(row, dict) for row in parsed):
        raise ValueError("NASA Exoplanet Archive response is not a row array")
    rows = []
    for row in parsed:
        if any(field not in row for field in FIELDS):
            raise ValueError("NASA Exoplanet Archive response schema is incomplete")
        rows.append({field: row[field] for field in FIELDS})
    return {"resultVersion": FREEZE_VERSION, "tapEndpoint": TAP_ENDPOINT, "exactEncodedQuery": encoded,
            "retrievalTimestamp": retrieved_at or datetime.now(timezone.utc).isoformat(),
            "httpStatus": status, "contentType": content_type, "returnedRows": rows,
            "rawResponseUTF8": raw.decode("utf-8"),
            "rawResponseSHA256": hashlib.sha256(raw).hexdigest(),
            "attributedSourceIdentifiers": {"ticID": tic, "gaiaDR3SourceID": gaia},
            "sourceLocalizationSHA256": review["sourceLocalizationSHA256"],
            "binaryConfirmationSHA256": review.get("binaryConfirmationSHA256"),
            "frozenEphemeris": review.get("frozenEphemeris"), "sourceAttributionReviewSHA256": sha256_json(review),
            "catalogAnswerKeyUsed": False}


def _number(value: Any) -> float | None:
    try: number = float(value)
    except (TypeError, ValueError): return None
    return number if math.isfinite(number) else None


def interpret_external_evidence(frozen: dict[str, Any]) -> dict[str, Any]:
    if frozen.get("resultVersion") != FREEZE_VERSION or frozen.get("catalogAnswerKeyUsed") is not False:
        raise ValueError("exact frozen external-evidence response is required")
    expected = frozen["attributedSourceIdentifiers"]
    tic, gaia = expected["ticID"], expected.get("gaiaDR3SourceID")
    period = _number((frozen.get("frozenEphemeris") or {}).get("refinedPeriodDays"))
    if period is None or period <= 0: raise ValueError("frozen ephemeris period is invalid")
    exact, conflicts = [], []
    for row in frozen.get("returnedRows") or []:
        if row.get("tic_id") != tic:
            conflicts.append(row); continue
        if gaia not in (None, "") and row.get("gaia_dr3_id") != gaia:
            conflicts.append(row); continue
        external = _number(row.get("pl_orbper"))
        error = max(abs(_number(row.get("pl_orbpererr1")) or 0), abs(_number(row.get("pl_orbpererr2")) or 0))
        if external is not None and abs(external - period) <= PERIOD_MATCH_ABSOLUTE_TOLERANCE_DAYS + error + 1e-12:
            exact.append(row)
    classification = "NO_MATCHING_EXTERNAL_COMPANION_EVIDENCE"; selected = None; regime = None
    if conflicts: classification = "CONFLICTING_EXTERNAL_COMPANION_EVIDENCE"
    elif len(exact) > 1: classification = "MULTIPLE_MATCHING_EXTERNAL_COMPANIONS"
    elif len(exact) == 1:
        selected = exact[0]
        mass, up, down = (_number(selected.get(key)) for key in ("pl_bmassj", "pl_bmassjerr1", "pl_bmassjerr2"))
        valid = (selected.get("default_flag") == 1 and selected.get("pl_controv_flag") == 0
                 and selected.get("rv_flag") == 1 and mass is not None and up is not None and down is not None
                 and up > 0 and down < 0 and selected.get("pl_bmassjlim") == 0
                 and bool(selected.get("pl_bmassprov")))
        if not valid:
            classification = "PERIOD_MATCHED_COMPANION_MASS_UNRESOLVED"
        else:
            low, high = mass + down, mass + up
            if high < 13: regime = "PLANETARY"; classification = "PERIOD_MATCHED_PLANETARY_MASS_COMPANION_SUPPORTED"
            elif low > 13 and high < 80: regime = "BROWN_DWARF"; classification = "PERIOD_MATCHED_BROWN_DWARF_MASS_COMPANION_SUPPORTED"
            elif low >= 80: regime = "STELLAR"; classification = "PERIOD_MATCHED_STELLAR_MASS_COMPANION_SUPPORTED"
            else: classification = "PERIOD_MATCHED_COMPANION_MASS_UNRESOLVED"
    ext_period = _number((selected or {}).get("pl_orbper")); mass = _number((selected or {}).get("pl_bmassj"))
    down = _number((selected or {}).get("pl_bmassjerr2")); up = _number((selected or {}).get("pl_bmassjerr1"))
    resolved = regime is not None
    return {"resultVersion": RESULT_VERSION, "classification": classification,
            "externalCompanionEvidenceResolved": resolved, "supportedCompanionMassRegime": regime,
            "externalOrbitalPeriodDays": ext_period,
            "externalOrbitalPeriodDifferenceDays": abs(ext_period-period) if ext_period is not None else None,
            "externalMassJupiter": mass,
            "externalMassIntervalJupiter": ([mass+down, mass+up] if None not in (mass, down, up) else None),
            "externalMassProvenance": (selected or {}).get("pl_bmassprov"), "selectedExternalRow": selected,
            "periodMatchAbsoluteToleranceDays": PERIOD_MATCH_ABSOLUTE_TOLERANCE_DAYS,
            "externalEvidenceMode": "PUBLISHED_COMPANION_CONFIRMATION",
            "externalKnownObjectCatalogUsed": True, "softwareBlindPhotometricEvidencePreserved": True,
            "catalogAnswerKeyUsed": False, "physicalMechanismResolved": False, "companionNatureResolved": False,
            "recommendedNextTest": ("FINAL_COMPANION_EVIDENCE_SYNTHESIS" if resolved
                                    else "EXTERNAL_COMPANION_EVIDENCE_REVIEW"),
            "externalEvidenceFreezeSHA256": sha256_json(frozen)}
