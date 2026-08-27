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


def canonical_tic_id(value: Any) -> str:
    """Return the archive's canonical TIC form without accepting loose prefixes."""
    if isinstance(value, bool):
        raise ValueError("malformed TIC identifier")
    if isinstance(value, int):
        number = value
    elif isinstance(value, str):
        text = value.strip()
        digits = text[4:] if text.startswith("TIC ") else text
        if not digits.isascii() or not digits.isdigit():
            raise ValueError("malformed TIC identifier")
        number = int(digits)
    else:
        raise ValueError("malformed TIC identifier")
    if number <= 0:
        raise ValueError("malformed TIC identifier")
    return f"TIC {number}"


def canonical_gaia_dr3_id(value: Any) -> str:
    """Return an exact Gaia DR3 identifier, rejecting other Gaia releases."""
    if isinstance(value, bool):
        raise ValueError("malformed Gaia DR3 identifier")
    if isinstance(value, int):
        number = value
    elif isinstance(value, str):
        text = value.strip()
        digits = text[9:] if text.startswith("Gaia DR3 ") else text
        if not digits.isascii() or not digits.isdigit():
            raise ValueError("malformed Gaia DR3 identifier")
        number = int(digits)
    else:
        raise ValueError("malformed Gaia DR3 identifier")
    if number <= 0:
        raise ValueError("malformed Gaia DR3 identifier")
    return f"Gaia DR3 {number}"


def review_source_attribution(localization: dict[str, Any]) -> dict[str, Any]:
    if not localization_gate(localization):
        raise ValueError("exact localization-v1 source-attribution gate is not satisfied")
    attributed = localization.get("attributedCatalogHypothesis")
    support, conflicts, ambiguous = [], [], []
    seen_sectors: set[Any] = set()
    duplicate_sectors = []
    top_class = localization["classification"]
    expected_sector_class = {
        "TARGET_CONSISTENT_ECLIPSE_SOURCE": "TARGET_CONSISTENT",
        "OFF_TARGET_CATALOG_CANDIDATE_ECLIPSE_SOURCE": "CATALOG_CANDIDATE_CONSISTENT",
        "CONSISTENTLY_OFF_TARGET_ECLIPSE_SOURCE": "OFF_CATALOG",
    }[top_class]
    for sector in localization.get("sectorResults") or []:
        if sector.get("role") != "INDEPENDENT":
            continue
        sector_id = sector.get("sector")
        if (isinstance(sector_id, bool) or not isinstance(sector_id, int)
                or sector_id <= 0 or sector_id in seen_sectors):
            duplicate_sectors.append(sector_id)
            continue
        seen_sectors.add(sector_id)
        key = _source_key(sector) if sector.get("usable") is True else None
        if key is None:
            ambiguous.append(sector_id)
        elif (key == attributed and sector.get("classification") == expected_sector_class
              and (top_class != "CONSISTENTLY_OFF_TARGET_ECLIPSE_SOURCE"
                   or sector.get("matchedCatalogHypothesis") is None)):
            support.append(sector_id)
        else:
            conflicts.append({"sector": sector_id, "source": key,
                              "classification": sector.get("classification")})
    catalog = localization.get("frozenCatalog") or {}
    hypotheses = catalog.get("catalogHypotheses") or []
    matches = [row for row in hypotheses if str(row.get("sourceID")) == str(attributed)]
    targets = [row for row in hypotheses if row.get("isTarget") is True]
    source_ids = [row.get("sourceID") for row in hypotheses]
    catalog_valid = (all(isinstance(value, str) and bool(value.strip()) for value in source_ids)
                     and len(source_ids) == len(set(source_ids)))
    off_catalog = attributed == "OFF_CATALOG_SKY_CLUSTER"
    role_consistent = (catalog_valid and isinstance(attributed, str) and bool(attributed.strip())
        and len(targets) == 1 and (
        (top_class == "TARGET_CONSISTENT_ECLIPSE_SOURCE" and len(matches) == 1
         and matches[0].get("isTarget") is True)
        or (top_class == "OFF_TARGET_CATALOG_CANDIDATE_ECLIPSE_SOURCE" and len(matches) == 1
            and matches[0].get("isTarget") is False)
        or (top_class == "CONSISTENTLY_OFF_TARGET_ECLIPSE_SOURCE" and off_catalog
            and not matches)))
    passed = (len(support) >= MIN_INDEPENDENT_SUPPORT and not conflicts
              and not duplicate_sectors and role_consistent)
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
            "duplicateIndependentSectors": duplicate_sectors,
            "primarySectorCanSatisfyReplication": False, "requiredIndependentSectorCount": MIN_INDEPENDENT_SUPPORT,
            "attributedSource": source, "attributedCatalogHypothesis": attributed,
            "sourceLocalizationSHA256": sha256_json(localization),
            "binaryConfirmationSHA256": localization.get("binaryConfirmationSHA256"),
            "frozenEphemeris": localization.get("frozenEphemeris"), "catalogAnswerKeyUsed": False,
            "physicalMechanismResolved": False, "companionNatureResolved": False,
            "recommendedNextTest": ("EXTERNAL_COMPANION_EVIDENCE_FREEZE" if classification.endswith("REVIEW_PASSED")
                                    else "CATALOG_IDENTITY_FOLLOWUP" if off_catalog
                                    else "ADDITIONAL_SPATIAL_EVIDENCE")}


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
    raw_tic = source.get("ticID")
    raw_gaia = source.get("gaiaDR3SourceID")
    tic = canonical_tic_id(raw_tic)
    gaia = canonical_gaia_dr3_id(raw_gaia) if raw_gaia not in (None, "") else None
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
            "attributedSourceIdentifiers": {"ticID": tic, "gaiaDR3SourceID": gaia,
                                            "rawTICID": raw_tic,
                                            "rawGaiaDR3SourceID": raw_gaia},
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
        try:
            row_tic = canonical_tic_id(row.get("tic_id"))
            row_gaia = (canonical_gaia_dr3_id(row.get("gaia_dr3_id"))
                        if gaia is not None else None)
        except ValueError:
            conflicts.append(row); continue
        if row_tic != tic:
            conflicts.append(row); continue
        if gaia is not None and row_gaia != gaia:
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
                 and mass > 0 and up > 0 and down < 0 and mass + down > 0
                 and selected.get("pl_bmassjlim") == 0
                 and isinstance(selected.get("pl_bmassprov"), str)
                 and bool(selected["pl_bmassprov"].strip()))
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
