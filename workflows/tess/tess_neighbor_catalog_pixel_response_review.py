"""Catalog-guided review of an unresolved residual-mode pixel response.

The v20.11 localization review already contains the distributed, fixed-frequency
pixel power maps.  This continuation freezes its method before provider access,
queries only a fixed target neighborhood in TIC and Gaia DR3, projects those
catalog positions through the persisted per-window sky Jacobians, and compares
their responses without rereading flux or launching distributed work.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from typing import Any, Callable

import numpy as np

from .tess_identity import _float, _int, _row_value, _separation_arcsec
from .tess_offset_source import _merge_catalog_candidates


HANDLER_ID = "openstar.tess.neighbor-catalog-pixel-response-review.analyze"
RESULT_VERSION = "openstar.tess-neighbor-catalog-pixel-response-review.v1"
METHOD_CONTRACT_ID = (
    "openstar.tess.neighbor-catalog-pixel-response-review."
    "frozen-power-map-catalog-projection.v1"
)

CATALOG_RADIUS_ARCSEC = 120.0
RESPONSE_KERNEL_SIGMA_PIXELS = 0.75
RESPONSE_KERNEL_RADIUS_PIXELS = 2.0
MIN_RESOLVABLE_SEPARATION_PIXELS = 1.0
MIN_WINNER_RESPONSE_RATIO = 1.25
MIN_INDEPENDENT_SECTORS = 3

TARGET_SUPPORTED = "TARGET_RESIDUAL_PIXEL_RESPONSE_SUPPORTED"
NEIGHBOR_SUPPORTED = "NEIGHBOR_RESIDUAL_PIXEL_RESPONSE_SUPPORTED"
MULTI_SOURCE = "MULTI_SOURCE_RESIDUAL_PIXEL_RESPONSE"
INCOMPLETE = "NEIGHBOR_CATALOG_PIXEL_RESPONSE_REVIEW_INCOMPLETE"
INCONCLUSIVE = "NEIGHBOR_CATALOG_PIXEL_RESPONSE_REVIEW_INCONCLUSIVE"


def _positive(value: Any) -> float | None:
    number = _float(value)
    return number if number is not None and number > 0.0 else None


def _finite_map(value: Any, rows: int, columns: int) -> np.ndarray | None:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if array.shape != (rows, columns):
        return None
    if np.any(~np.isfinite(array)) or np.any(array < 0.0):
        return None
    return array


def method_contract_hash(contract: dict[str, Any]) -> str:
    encoded = json.dumps(
        contract, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_method_contract(
    *,
    localization_review: dict[str, Any],
    mode_identification: dict[str, Any],
) -> dict[str, Any]:
    """Freeze every query, scoring, and decision rule before provider access."""
    cross_time = localization_review.get("crossTime") or {}
    mode_candidate = mode_identification.get("modeCandidate") or {}
    return {
        "methodContractID": METHOD_CONTRACT_ID,
        "resultVersion": RESULT_VERSION,
        "evidenceBoundary": {
            "localizationReviewVersion": localization_review.get("version"),
            "localizationClassification": cross_time.get("classification"),
            "residualModeFrequencyCyclesPerDay": mode_candidate.get(
                "frequencyCyclesPerDay"
            ),
            "residualModePeriodDays": mode_candidate.get("periodDays"),
            "independentSectors": list(
                (mode_identification.get("independentSectorSupport") or {}).get(
                    "sectors"
                ) or []
            ),
        },
        "catalogAcquisition": {
            "sources": ["TIC", "GAIA_DR3"],
            "queryCenter": "PERSISTED_TARGET_SKY",
            "queryRadiusArcsec": CATALOG_RADIUS_ARCSEC,
            "queryOrderAffectsDecision": False,
            "targetCatalogAliasesAreExcludedFromNeighborCandidates": True,
        },
        "pixelResponseScoring": {
            "input": "PERSISTED_V20_11_CANDIDATE_POWER_MAPS_ONLY",
            "fluxValuesRead": False,
            "catalogProjection": "PERSISTED_PER_WINDOW_PIXEL_TO_SKY_JACOBIAN",
            "kernel": "GAUSSIAN_WEIGHTED_MEAN",
            "kernelSigmaPixels": RESPONSE_KERNEL_SIGMA_PIXELS,
            "kernelRadiusPixels": RESPONSE_KERNEL_RADIUS_PIXELS,
            "minimumResolvableSeparationPixels": (
                MIN_RESOLVABLE_SEPARATION_PIXELS
            ),
            "minimumWinnerResponseRatio": MIN_WINNER_RESPONSE_RATIO,
            "upstreamLocalizationQualityGateIsRetained": True,
        },
        "aggregateDecision": {
            "minimumIndependentSectors": MIN_INDEPENDENT_SECTORS,
            "sectorSupport": "STRICT_MAJORITY_OF_QUALITY_WINDOWS",
            "positiveAttributionRequiresCompleteTicAndGaiaQueries": True,
            "closeNeighborBlocksTargetOnlyAttribution": True,
            "physicalMechanismResolved": False,
            "claimLevelChanged": False,
        },
    }


def validate_review_boundary(
    *,
    preparation: dict[str, Any],
    localization_review: dict[str, Any],
    mode_identification: dict[str, Any],
    identity: dict[str, Any],
    expected_tic_id: int,
) -> None:
    """Fail closed unless the exact unresolved v20.11 lineage is intact."""
    cross_time = localization_review.get("crossTime") or {}
    support = mode_identification.get("independentSectorSupport") or {}
    mode_candidate = mode_identification.get("modeCandidate") or {}
    target_sky = localization_review.get("targetSky") or {}
    prepared_target_sky = preparation.get("targetSky") or {}
    frequency = _positive(mode_candidate.get("frequencyCyclesPerDay"))
    period = _positive(mode_candidate.get("periodDays"))
    review_frequency = _positive(
        localization_review.get("residualFrequencyAtReference")
    )
    preparation_frequency = _positive(
        preparation.get("residualFrequencyAtReference")
    )
    try:
        expected_tic = int(expected_tic_id)
        identity_tic = int(identity.get("ticID"))
        review_tic = int(localization_review.get("ticID"))
        preparation_tic = int(preparation.get("ticID"))
        sectors = sorted(int(value) for value in support.get("sectors") or [])
        signal_sectors = sorted(
            int(value) for value in localization_review.get("signalSectors") or []
        )
        preparation_sectors = sorted(
            int(value) for value in preparation.get("signalSectors") or []
        )
        eligible = int(cross_time.get("independentEligibleSectorCount"))
        required = int(cross_time.get("requiredIndependentSupportCount"))
        target_ra = float(target_sky["raDeg"])
        target_dec = float(target_sky["decDeg"])
        prepared_ra = float(prepared_target_sky["raDeg"])
        prepared_dec = float(prepared_target_sky["decDeg"])
    except (KeyError, TypeError, ValueError):
        raise RuntimeError(
            "Neighbor catalog/pixel-response lineage is incomplete."
        ) from None

    metadata_by_key = {
        str(item.get("windowKey")): item
        for item in preparation.get("windowMetadata") or []
        if isinstance(item, dict) and item.get("windowKey")
    }
    review_windows = localization_review.get("windowResults") or []
    window_lineage_valid = bool(review_windows and metadata_by_key)
    independent_quality_sectors: set[int] = set()
    for window in review_windows:
        if not isinstance(window, dict):
            window_lineage_valid = False
            break
        metadata = metadata_by_key.get(str(window.get("windowKey")))
        shape = window.get("shape") or []
        if metadata is None or len(shape) != 2:
            window_lineage_valid = False
            break
        try:
            rows, columns = int(shape[0]), int(shape[1])
            sector = int(window.get("sector"))
        except (TypeError, ValueError):
            window_lineage_valid = False
            break
        if not (
            rows > 0
            and columns > 0
            and list(metadata.get("shape") or []) == [rows, columns]
            and int(metadata.get("sector")) == sector
            and int(metadata.get("windowIndex")) == int(window.get("windowIndex"))
            and metadata.get("role") == window.get("role")
            and metadata.get("targetPixel") == window.get("targetPixel")
            and _finite_map(window.get("candidatePowerMap"), rows, columns)
            is not None
        ):
            window_lineage_valid = False
            break
        jacobian = metadata.get("skyJacobian") or {}
        if not all(
            _float(jacobian.get(key)) is not None
            for key in (
                "xToEastArcsec",
                "xToNorthArcsec",
                "yToEastArcsec",
                "yToNorthArcsec",
            )
        ):
            window_lineage_valid = False
            break
        if (
            window.get("role") == "independent"
            and window.get("localizationQualityPass") is True
        ):
            independent_quality_sectors.add(sector)

    exact = (
        preparation.get("available") is True
        and localization_review.get("version")
        == "openstar.tess-residual-mode-source-localization-review.v1"
        and localization_review.get("recommendedNextTest")
        == "NEIGHBOR_CATALOG_AND_PIXEL_RESPONSE_REVIEW"
        and localization_review.get("physicalMechanismResolved") is False
        and localization_review.get("claimLevelChanged") is False
        and cross_time.get("classification")
        == "RESIDUAL_MODE_TIME_RESOLVED_LOCALIZATION_UNRESOLVED"
        and cross_time.get("residualModeOrigin") == "UNRESOLVED"
        and cross_time.get("recommendedNextTest")
        == "NEIGHBOR_CATALOG_AND_PIXEL_RESPONSE_REVIEW"
        and mode_identification.get("classification") == "INDEPENDENT_STABLE_MODE"
        and mode_identification.get("independentModeEvidenceSurvived") is True
        and mode_identification.get("physicalMechanismResolved") is False
        and isinstance(mode_identification.get("modeCandidate"), dict)
        and support.get("sufficient") is True
        and len(sectors) >= MIN_INDEPENDENT_SECTORS
        and int(support.get("count") or 0) == len(sectors)
        and int(support.get("requiredCount") or 0) >= MIN_INDEPENDENT_SECTORS
        and eligible >= MIN_INDEPENDENT_SECTORS
        and required >= MIN_INDEPENDENT_SECTORS
        and set(independent_quality_sectors).issubset(set(sectors))
        and len(independent_quality_sectors) >= MIN_INDEPENDENT_SECTORS
        and set(sectors).issubset(set(signal_sectors))
        and signal_sectors == preparation_sectors
        and frequency is not None
        and period is not None
        and review_frequency is not None
        and preparation_frequency is not None
        and math.isclose(frequency, review_frequency, rel_tol=1e-9, abs_tol=1e-12)
        and math.isclose(
            frequency, preparation_frequency, rel_tol=1e-9, abs_tol=1e-12
        )
        and math.isclose(period, 1.0 / frequency, rel_tol=1e-9, abs_tol=1e-12)
        and math.isclose(target_ra, prepared_ra, rel_tol=0.0, abs_tol=1e-10)
        and math.isclose(target_dec, prepared_dec, rel_tol=0.0, abs_tol=1e-10)
        and expected_tic == identity_tic == review_tic == preparation_tic
        and identity.get("identityResolved") is True
        and (identity.get("tic") or {}).get("found") is True
        and window_lineage_valid
    )
    if not exact:
        raise RuntimeError(
            "Neighbor catalog/pixel-response review requires the exact unresolved "
            "v20.11 independent stable-mode localization boundary."
        )


def _query_neighbor_catalogs(
    target_sky: dict[str, float], target_tic_id: int
) -> dict[str, Any]:
    """Acquire a bounded catalog snapshot; callers persist the canonical result."""
    from astropy import units as u
    from astropy.coordinates import SkyCoord
    from astroquery.mast import Catalogs
    from astroquery.vizier import Vizier

    coordinate = SkyCoord(
        float(target_sky["raDeg"]) * u.deg,
        float(target_sky["decDeg"]) * u.deg,
        frame="icrs",
    )
    tic: dict[str, Any] = {"sources": []}
    gaia: dict[str, Any] = {"sources": []}
    try:
        table = Catalogs.query_region(
            coordinate,
            radius=CATALOG_RADIUS_ARCSEC / 3600.0,
            catalog="TIC",
        )
        for row in ([] if table is None else table):
            tic_id = _int(_row_value(row, ("ID", "id")))
            ra = _float(_row_value(row, ("ra", "RA")))
            dec = _float(_row_value(row, ("dec", "DEC")))
            if tic_id is None or ra is None or dec is None:
                continue
            source = SkyCoord(ra * u.deg, dec * u.deg, frame="icrs")
            tic["sources"].append({
                "catalog": "TIC",
                "ticID": int(tic_id),
                "isTargetTIC": int(tic_id) == int(target_tic_id),
                "gaiaSourceID": _int(_row_value(row, ("GAIA", "Gaia"))),
                "raDeg": float(ra),
                "decDeg": float(dec),
                "separationArcsec": float(coordinate.separation(source).arcsec),
                "tmag": _float(_row_value(row, ("Tmag",))),
            })
        tic["sources"].sort(
            key=lambda item: (float(item["separationArcsec"]), int(item["ticID"]))
        )
    except Exception as error:
        tic["queryError"] = f"{type(error).__name__}: {error}"

    try:
        result = Vizier(columns=["*", "+_r"], row_limit=500).query_region(
            coordinate,
            radius=CATALOG_RADIUS_ARCSEC * u.arcsec,
            catalog="I/355/gaiadr3",
        )
        table = result[0] if len(result) else []
        for row in table:
            source_id = _int(_row_value(row, ("Source", "source_id")))
            ra = _float(_row_value(row, ("RA_ICRS", "RAJ2000", "ra")))
            dec = _float(_row_value(row, ("DE_ICRS", "DEJ2000", "dec")))
            if source_id is None or ra is None or dec is None:
                continue
            separation = _separation_arcsec(table, row)
            if separation is None:
                source = SkyCoord(ra * u.deg, dec * u.deg, frame="icrs")
                separation = float(coordinate.separation(source).arcsec)
            gaia["sources"].append({
                "catalog": "GaiaDR3",
                "gaiaSourceID": int(source_id),
                "raDeg": float(ra),
                "decDeg": float(dec),
                "separationArcsec": float(separation),
                "gMag": _float(_row_value(row, ("Gmag", "phot_g_mean_mag"))),
                "bpMag": _float(_row_value(row, ("BPmag", "phot_bp_mean_mag"))),
                "rpMag": _float(_row_value(row, ("RPmag", "phot_rp_mean_mag"))),
            })
        gaia["sources"].sort(
            key=lambda item: (
                float(item["separationArcsec"]), int(item["gaiaSourceID"])
            )
        )
    except Exception as error:
        gaia["queryError"] = f"{type(error).__name__}: {error}"
    return {"tic": tic, "gaiaDR3": gaia}


def _target_gaia_ids(identity: dict[str, Any]) -> set[int]:
    values = [
        ((identity.get("gaiaDR3") or {}).get("nearest") or {}).get("sourceID"),
        ((identity.get("tic") or {}).get("aliases") or {}).get("GAIA_field"),
    ]
    return {value for item in values if (value := _int(item)) is not None}


def _catalog_candidates(
    snapshot: dict[str, Any], identity: dict[str, Any], target_sky: dict[str, float]
) -> list[dict[str, Any]]:
    tic_result = snapshot.get("tic")
    gaia_result = snapshot.get("gaiaDR3")
    tic_sources = (
        tic_result.get("sources")
        if isinstance(tic_result, dict)
        and isinstance(tic_result.get("sources"), list)
        else []
    )
    gaia_sources = (
        gaia_result.get("sources")
        if isinstance(gaia_result, dict)
        and isinstance(gaia_result.get("sources"), list)
        else []
    )
    candidates = _merge_catalog_candidates(
        tic_sources=list(tic_sources),
        gaia_sources=list(gaia_sources),
        target_sky=target_sky,
        exclude_target_neighborhood=False,
    )
    target_gaia = _target_gaia_ids(identity)
    canonical = []
    for item in candidates:
        gaia_id = _int(((item.get("gaiaDR3") or {}).get("gaiaSourceID")))
        tic_id = _int(((item.get("tic") or {}).get("ticID")))
        if gaia_id is not None and gaia_id in target_gaia:
            continue
        stable_id = (
            f"GAIA_DR3:{gaia_id}" if gaia_id is not None else f"TIC:{tic_id}"
        )
        canonical.append({**item, "sourceID": stable_id})
    canonical.sort(
        key=lambda item: (
            float(item.get("targetSeparationArcsec") or float("inf")),
            str(item["sourceID"]),
        )
    )
    for index, item in enumerate(canonical, start=1):
        item["candidateRank"] = index
    return canonical


def _catalog_query_errors(snapshot: dict[str, Any]) -> list[str]:
    errors = []
    for key, label in (("tic", "TIC"), ("gaiaDR3", "GaiaDR3")):
        result = snapshot.get(key)
        if not isinstance(result, dict):
            errors.append(f"{label}: query result is missing or malformed")
        elif result.get("queryError"):
            errors.append(f"{label}: {result.get('queryError')}")
        elif not isinstance(result.get("sources"), list):
            errors.append(f"{label}: query source list is missing or malformed")
    return errors


def _sky_offsets_arcsec(
    target_sky: dict[str, float], source: dict[str, Any]
) -> tuple[float, float]:
    from astropy import units as u
    from astropy.coordinates import SkyCoord

    target = SkyCoord(
        float(target_sky["raDeg"]) * u.deg,
        float(target_sky["decDeg"]) * u.deg,
        frame="icrs",
    )
    candidate = SkyCoord(
        float(source["raDeg"]) * u.deg,
        float(source["decDeg"]) * u.deg,
        frame="icrs",
    )
    east, north = target.spherical_offsets_to(candidate)
    return float(east.arcsec), float(north.arcsec)


def _project_pixel(
    *,
    target_pixel: dict[str, Any],
    jacobian: dict[str, Any],
    east_arcsec: float,
    north_arcsec: float,
) -> tuple[float, float] | None:
    matrix = np.asarray([
        [_float(jacobian.get("xToEastArcsec")), _float(jacobian.get("yToEastArcsec"))],
        [_float(jacobian.get("xToNorthArcsec")), _float(jacobian.get("yToNorthArcsec"))],
    ], dtype=np.float64)
    if np.any(~np.isfinite(matrix)) or abs(float(np.linalg.det(matrix))) < 1e-9:
        return None
    delta = np.linalg.solve(matrix, np.asarray([east_arcsec, north_arcsec]))
    x = _float(target_pixel.get("x"))
    y = _float(target_pixel.get("y"))
    if x is None or y is None:
        return None
    return float(x + delta[0]), float(y + delta[1])


def _response_score(power_map: np.ndarray, x: float, y: float) -> float | None:
    rows, columns = power_map.shape
    if not (-0.5 <= x <= columns - 0.5 and -0.5 <= y <= rows - 0.5):
        return None
    yy, xx = np.indices(power_map.shape, dtype=np.float64)
    distance = np.hypot(xx - float(x), yy - float(y))
    mask = distance <= RESPONSE_KERNEL_RADIUS_PIXELS
    if not np.any(mask):
        return None
    weights = np.exp(-0.5 * np.square(distance / RESPONSE_KERNEL_SIGMA_PIXELS))
    weights = np.where(mask, weights, 0.0)
    total = float(np.sum(weights))
    return float(np.sum(power_map * weights) / total) if total > 0.0 else None


def _window_evidence(
    *,
    window: dict[str, Any],
    metadata: dict[str, Any],
    candidates: list[dict[str, Any]],
    target_sky: dict[str, float],
) -> dict[str, Any]:
    rows, columns = (int(value) for value in window["shape"])
    power_map = _finite_map(window["candidatePowerMap"], rows, columns)
    assert power_map is not None
    target_pixel = metadata["targetPixel"]
    target_x = float(target_pixel["x"])
    target_y = float(target_pixel["y"])
    scores = [{
        "sourceID": "TARGET",
        "sourceType": "TARGET",
        "pixel": {"x": target_x, "y": target_y},
        "targetSeparationPixels": 0.0,
        "responseScore": _response_score(power_map, target_x, target_y),
    }]
    close_neighbors = []
    for candidate in candidates:
        east, north = _sky_offsets_arcsec(target_sky, candidate)
        pixel = _project_pixel(
            target_pixel=target_pixel,
            jacobian=metadata["skyJacobian"],
            east_arcsec=east,
            north_arcsec=north,
        )
        if pixel is None:
            continue
        x, y = pixel
        score = _response_score(power_map, x, y)
        if score is None:
            continue
        separation_pixels = math.hypot(x - target_x, y - target_y)
        if separation_pixels < MIN_RESOLVABLE_SEPARATION_PIXELS:
            close_neighbors.append(candidate["sourceID"])
        scores.append({
            "sourceID": candidate["sourceID"],
            "sourceType": "NEIGHBOR",
            "pixel": {"x": x, "y": y},
            "targetSeparationPixels": separation_pixels,
            "responseScore": score,
        })
    ranked = sorted(
        (item for item in scores if item.get("responseScore") is not None),
        key=lambda item: (-float(item["responseScore"]), str(item["sourceID"])),
    )
    winner = ranked[0] if ranked else None
    runner_up = ranked[1] if len(ranked) > 1 else None
    ratio = None
    if winner is not None:
        denominator = float((runner_up or {}).get("responseScore") or 0.0)
        ratio = (
            float(winner["responseScore"]) / denominator
            if denominator > 0.0 else None
        )
    classification = "AMBIGUOUS"
    source_id = None
    decisive = winner is not None and (
        runner_up is None
        or (ratio is not None and ratio >= MIN_WINNER_RESPONSE_RATIO)
    )
    if decisive:
        if winner["sourceID"] == "TARGET" and close_neighbors:
            classification = "UNRESOLVED_TARGET_NEIGHBOR_BLEND"
        elif winner["sourceID"] == "TARGET":
            classification = "TARGET"
            source_id = "TARGET"
        elif float(winner["targetSeparationPixels"]) >= MIN_RESOLVABLE_SEPARATION_PIXELS:
            classification = "NEIGHBOR"
            source_id = str(winner["sourceID"])
    return {
        "windowKey": window.get("windowKey"),
        "sector": int(window["sector"]),
        "windowIndex": int(window["windowIndex"]),
        "upstreamClassification": window.get("classification"),
        "upstreamLocalizationQualityPass": window.get("localizationQualityPass"),
        "classification": classification,
        "supportedSourceID": source_id,
        "winnerResponseRatio": ratio,
        "closeUnresolvedNeighborSourceIDs": sorted(close_neighbors),
        "sourceResponses": ranked,
    }


def _sector_evidence(windows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_sector: dict[int, list[dict[str, Any]]] = {}
    for window in windows:
        by_sector.setdefault(int(window["sector"]), []).append(window)
    results = []
    for sector in sorted(by_sector):
        rows = by_sector[sector]
        support = Counter(
            str(row["supportedSourceID"])
            for row in rows if row.get("supportedSourceID")
        )
        classification = "AMBIGUOUS"
        source_id = None
        if support:
            candidate, count = sorted(
                support.items(), key=lambda item: (-item[1], item[0])
            )[0]
            if count > len(rows) / 2.0:
                source_id = candidate
                classification = "TARGET" if candidate == "TARGET" else "NEIGHBOR"
        results.append({
            "sector": sector,
            "qualityWindowCount": len(rows),
            "classification": classification,
            "supportedSourceID": source_id,
            "windowSupportCounts": dict(sorted(support.items())),
        })
    return results


def analyze_neighbor_catalog_pixel_response_review(
    *,
    preparation: dict[str, Any],
    localization_review: dict[str, Any],
    mode_identification: dict[str, Any],
    identity: dict[str, Any],
    expected_tic_id: int,
    catalog_provider: Callable[[dict[str, float], int], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    validate_review_boundary(
        preparation=preparation,
        localization_review=localization_review,
        mode_identification=mode_identification,
        identity=identity,
        expected_tic_id=expected_tic_id,
    )
    contract = build_method_contract(
        localization_review=localization_review,
        mode_identification=mode_identification,
    )
    contract_hash = method_contract_hash(contract)
    target_sky = dict(localization_review["targetSky"])
    provider = catalog_provider or _query_neighbor_catalogs
    snapshot = provider(target_sky, int(expected_tic_id))
    if not isinstance(snapshot, dict):
        raise RuntimeError("Neighbor catalog provider returned a malformed snapshot.")
    candidates = _catalog_candidates(snapshot, identity, target_sky)
    query_errors = _catalog_query_errors(snapshot)
    catalog_complete = not query_errors

    metadata_by_key = {
        str(item["windowKey"]): item
        for item in preparation["windowMetadata"]
    }
    window_evidence = []
    for window in localization_review["windowResults"]:
        if not (
            window.get("role") == "independent"
            and window.get("localizationQualityPass") is True
        ):
            continue
        window_evidence.append(_window_evidence(
            window=window,
            metadata=metadata_by_key[str(window["windowKey"])],
            candidates=candidates,
            target_sky=target_sky,
        ))
    sector_evidence = _sector_evidence(window_evidence)
    target_sectors = sorted(
        item["sector"] for item in sector_evidence
        if item["supportedSourceID"] == "TARGET"
    )
    neighbor_support: dict[str, list[int]] = {}
    for item in sector_evidence:
        source_id = item.get("supportedSourceID")
        if source_id and source_id != "TARGET":
            neighbor_support.setdefault(str(source_id), []).append(int(item["sector"]))
    neighbor_support = {
        key: sorted(value) for key, value in sorted(neighbor_support.items())
    }
    best_neighbor_id = None
    best_neighbor_sectors: list[int] = []
    if neighbor_support:
        best_neighbor_id, best_neighbor_sectors = sorted(
            neighbor_support.items(), key=lambda item: (-len(item[1]), item[0])
        )[0]

    if not catalog_complete:
        classification = INCOMPLETE
        next_test = "RETRY_NEIGHBOR_CATALOG_AND_PIXEL_RESPONSE_REVIEW"
    elif (
        len(target_sectors) >= MIN_INDEPENDENT_SECTORS
        and len(best_neighbor_sectors) < MIN_INDEPENDENT_SECTORS
    ):
        classification = TARGET_SUPPORTED
        next_test = "EXTERNAL_VARIABILITY_CLASSIFICATION_AND_BINARY_EVIDENCE"
    elif (
        len(best_neighbor_sectors) >= MIN_INDEPENDENT_SECTORS
        and len(target_sectors) < MIN_INDEPENDENT_SECTORS
    ):
        classification = NEIGHBOR_SUPPORTED
        next_test = "NEIGHBOR_RESIDUAL_VARIABILITY_VALIDATION"
    elif target_sectors and best_neighbor_sectors:
        classification = MULTI_SOURCE
        next_test = "MULTI_SOURCE_RESIDUAL_DECOMPOSITION"
    else:
        classification = INCONCLUSIVE
        next_test = "HUMAN_SCIENTIFIC_REVIEW"

    return {
        "version": RESULT_VERSION,
        "methodContractID": METHOD_CONTRACT_ID,
        "methodContractHash": contract_hash,
        "methodContract": contract,
        "ticID": int(expected_tic_id),
        "targetSky": target_sky,
        "residualMode": dict(mode_identification["modeCandidate"]),
        "catalogSnapshot": snapshot,
        "catalogQueryComplete": catalog_complete,
        "catalogQueryErrors": query_errors,
        "catalogCandidates": candidates,
        "windowEvidence": window_evidence,
        "sectorEvidence": sector_evidence,
        "aggregateDecision": {
            "minimumIndependentSectors": MIN_INDEPENDENT_SECTORS,
            "targetSupportingSectors": target_sectors,
            "neighborSupportingSectorsBySource": neighbor_support,
            "bestNeighborSourceID": best_neighbor_id,
            "bestNeighborSupportingSectors": best_neighbor_sectors,
        },
        "classification": classification,
        "residualModeOrigin": (
            "TARGET_CONSISTENT" if classification == TARGET_SUPPORTED
            else "NEIGHBOR" if classification == NEIGHBOR_SUPPORTED
            else "TIME_VARIABLE_OR_BLENDED" if classification == MULTI_SOURCE
            else "UNRESOLVED"
        ),
        "claimLevelChanged": False,
        "physicalMechanismResolved": False,
        "recommendedNextTest": next_test,
        "interpretationGuard": (
            "This review attributes only the independently stable residual mode. "
            "It does not resolve the physical mechanism, upgrade the claim, or "
            "alter the established periodic-family interpretation."
        ),
    }
