import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astroquery.vizier import Vizier

PROJECT_ID = "openstar.tess-blind-published-v3"
LOCK_PATH = Path("data/projects/openstar.tess-blind-published-v3.lock.json")
OUTPUT_PATH = Path("data/projects/openstar.tess-blind-published-v3.reveal-v1.json")
VSX_CATALOG = "B/vsx/vsx"
VSX_QUERY_RADIUS_ARCSEC = 12.0
PERIOD_MIN_DAYS = 0.2
PERIOD_MAX_DAYS = 10.0
MAX_SEPARATION_DIFFERENCE_ARCSEC = 0.75
MIN_MATCH_ADVANTAGE_ARCSEC = 0.20

# Frozen from the completed coordinator run BEFORE catalog reveal.
FROZEN_RESULTS = {
    "Blind V2-A": {"ticID": 149107258, "openstarFrequency": 0.32640239, "openstarPeriodDays": 3.06370306, "openstarPower": 0.97203618, "astropyFrequency": 0.32632177, "astropyPeriodDays": 3.06445998, "astropyPower": 0.97802784, "hardFailedWorkUnits": 0},
    "Blind V2-B": {"ticID": 356108440, "openstarFrequency": 3.74863136, "openstarPeriodDays": 0.26676403, "openstarPower": 0.75823838, "astropyFrequency": 3.74861505, "astropyPeriodDays": 0.26676519, "astropyPower": 0.75825743, "hardFailedWorkUnits": 0},
    "Blind V2-C": {"ticID": 408350711, "openstarFrequency": 0.25696622, "openstarPeriodDays": 3.89156207, "openstarPower": 0.72754276, "astropyFrequency": 0.25690780, "astropyPeriodDays": 3.89244707, "astropyPower": 0.72892570, "hardFailedWorkUnits": 0},
    "Blind V2-D": {"ticID": 233684019, "openstarFrequency": 0.27349111, "openstarPeriodDays": 3.65642600, "openstarPower": 0.15778162, "astropyFrequency": 0.27353084, "astropyPeriodDays": 3.65589488, "astropyPower": 0.15785015, "hardFailedWorkUnits": 1},
    "Blind V2-E": {"ticID": 233679640, "openstarFrequency": 0.45423007, "openstarPeriodDays": 2.20152752, "openstarPower": 0.44213399, "astropyFrequency": 0.45419269, "astropyPeriodDays": 2.20170873, "astropyPower": 0.44234581, "hardFailedWorkUnits": 0},
    "Blind V2-F": {"ticID": 315229214, "openstarFrequency": 0.26673746, "openstarPeriodDays": 3.74900472, "openstarPower": 0.36782596, "astropyFrequency": 0.26674213, "astropyPeriodDays": 3.74893904, "astropyPower": 0.36791904, "hardFailedWorkUnits": 2},
    "Blind V2-G": {"ticID": 164697828, "openstarFrequency": 0.15263672, "openstarPeriodDays": 6.55150339, "openstarPower": 0.00152596, "astropyFrequency": 0.15073364, "astropyPeriodDays": 6.63421924, "astropyPower": 0.00157528, "hardFailedWorkUnits": 1},
    "Blind V2-H": {"ticID": 349231109, "openstarFrequency": 4.89004169, "openstarPeriodDays": 0.20449723, "openstarPower": 0.97587776, "astropyFrequency": 4.89003603, "astropyPeriodDays": 0.20449747, "astropyPower": 0.97591061, "hardFailedWorkUnits": 0},
}


def python_value(value):
    if value is None or np.ma.is_masked(value):
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def float_or_none(value):
    value = python_value(value)
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def int_or_none(value):
    value = python_value(value)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None


def row_value(row, names, default=None):
    colnames = set(getattr(row, "colnames", []))
    for name in names:
        if name not in colnames:
            continue
        try:
            value = python_value(row[name])
        except Exception:
            continue
        if value is not None:
            return value
    return default


def canonical_payload(lock_document):
    payload = dict(lock_document)
    payload.pop("lockSHA256", None)
    return payload


def calculate_lock_sha256(lock_document):
    encoded = json.dumps(
        canonical_payload(lock_document),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def vsx_coordinate_from_row(row):
    ra = row_value(row, ("RAJ2000", "RA_ICRS"))
    dec = row_value(row, ("DEJ2000", "DE_ICRS"))
    if ra is None or dec is None:
        return None
    try:
        return SkyCoord(float(ra), float(dec), unit="deg", frame="icrs")
    except (TypeError, ValueError):
        try:
            return SkyCoord(str(ra), str(dec), unit=(u.hourangle, u.deg), frame="icrs")
        except Exception:
            return None


def period_percent_error(measured, published):
    return abs(measured - published) / abs(published) * 100.0


def frequency_percent_error(measured_frequency, published_period):
    published_frequency = 1.0 / published_period
    return abs(measured_frequency - published_frequency) / published_frequency * 100.0


def harmonic_relation(measured_period, published_period):
    ratio = measured_period / published_period
    candidates = (
        (0.25, "1/4× published period"),
        (0.50, "1/2× published period"),
        (1.00, "direct period"),
        (2.00, "2× published period"),
        (3.00, "3× published period"),
        (4.00, "4× published period"),
    )
    expected, label = min(candidates, key=lambda item: abs(ratio - item[0]))
    relation_error_percent = abs(ratio - expected) / expected * 100.0
    return {"label": label, "ratio": ratio, "relationErrorPercent": relation_error_percent}


def query_vsx_candidates(target):
    coordinate = SkyCoord(float(target["ra"]), float(target["dec"]), unit="deg", frame="icrs")
    vizier = Vizier(
        columns=["OID", "Name", "Type", "Period", "RAJ2000", "DEJ2000"],
        row_limit=50,
    )
    result = vizier.query_region(
        coordinate,
        radius=VSX_QUERY_RADIUS_ARCSEC * u.arcsec,
        catalog=VSX_CATALOG,
    )
    if len(result) == 0:
        return []

    candidates = []
    for row in result[0]:
        period = float_or_none(row_value(row, ("Period", "period")))
        if period is None or period < PERIOD_MIN_DAYS or period > PERIOD_MAX_DAYS:
            continue

        vsx_coordinate = vsx_coordinate_from_row(row)
        if vsx_coordinate is None:
            continue

        separation = float(coordinate.separation(vsx_coordinate).arcsec)
        candidates.append({
            "oid": int_or_none(row_value(row, ("OID",))),
            "name": row_value(row, ("Name", "name")),
            "type": row_value(row, ("Type", "type")),
            "periodDays": period,
            "separationArcsec": separation,
            "ra": float(vsx_coordinate.ra.deg),
            "dec": float(vsx_coordinate.dec.deg),
        })

    candidates.sort(key=lambda item: (item["separationArcsec"], item["periodDays"]))
    return candidates


def choose_preregistered_vsx_match(target, candidates):
    if not candidates:
        return {
            "matched": False,
            "reason": "No period-bearing VSX source was found in the reveal cone.",
        }

    expected = float(target["vsxToTicSeparationArcsec"])
    ranked = sorted(candidates, key=lambda item: abs(item["separationArcsec"] - expected))
    best = ranked[0]
    best_difference = abs(best["separationArcsec"] - expected)

    if best_difference > MAX_SEPARATION_DIFFERENCE_ARCSEC:
        return {
            "matched": False,
            "reason": "Reveal candidate does not reproduce preregistered VSX-to-TIC separation.",
            "expectedSeparationArcsec": expected,
            "bestDifferenceArcsec": best_difference,
            "candidates": candidates,
        }

    if len(ranked) > 1:
        second_difference = abs(ranked[1]["separationArcsec"] - expected)
        if second_difference - best_difference < MIN_MATCH_ADVANTAGE_ARCSEC:
            return {
                "matched": False,
                "reason": "VSX reveal match is ambiguous between multiple period-bearing sources.",
                "expectedSeparationArcsec": expected,
                "candidates": candidates,
            }

    return {
        "matched": True,
        "expectedSeparationArcsec": expected,
        "separationDifferenceArcsec": best_difference,
        "source": best,
        "candidateCount": len(candidates),
    }


def comparison(period, frequency, published_period):
    return {
        "periodDays": period,
        "frequency": frequency,
        "periodErrorDays": abs(period - published_period),
        "periodErrorPercent": period_percent_error(period, published_period),
        "frequencyErrorPercent": frequency_percent_error(frequency, published_period),
        "harmonicRelation": harmonic_relation(period, published_period),
    }


def print_comparison(label, value):
    print(f"   {label} period: {value['periodDays']:.8f} d")
    print(
        f"   {label} period error: {value['periodErrorDays']:.8f} d "
        f"({value['periodErrorPercent']:.4f}%)"
    )
    print(
        f"   {label} relation: {value['harmonicRelation']['label']} "
        f"(relation error {value['harmonicRelation']['relationErrorPercent']:.4f}%)"
    )


def reveal_target(target):
    blind_name = target["blindName"]
    frozen = FROZEN_RESULTS[blind_name]

    if int(target["ticID"]) != int(frozen["ticID"]):
        raise RuntimeError(f"Frozen TIC mismatch for {blind_name}.")

    print()
    print("════════════════════════════════════════════════════════")
    print(f"⭐ {blind_name} — TIC {target['ticID']}")
    print("════════════════════════════════════════════════════════")
    print(f"   frozen OpenStar period: {frozen['openstarPeriodDays']:.8f} d")
    print(f"   frozen Astropy period: {frozen['astropyPeriodDays']:.8f} d")
    print(f"   hard-failed work units: {frozen['hardFailedWorkUnits']}")
    print(
        "   preregistered VSX↔TIC separation: "
        f"{float(target['vsxToTicSeparationArcsec']):.4f} arcsec"
    )

    try:
        candidates = query_vsx_candidates(target)
    except Exception as error:
        print()
        print(f"❌ VSX QUERY ERROR: {type(error).__name__}: {error}")
        return {
            "blindName": blind_name,
            "ticID": int(target["ticID"]),
            "frozen": frozen,
            "status": "VSX QUERY ERROR",
            "error": f"{type(error).__name__}: {error}",
        }

    match = choose_preregistered_vsx_match(target, candidates)

    if not match["matched"]:
        print()
        print("❌ REVEAL MATCH FAILED")
        print(f"   reason: {match['reason']}")
        print(f"   candidate period-bearing VSX sources: {len(candidates)}")
        for candidate in candidates:
            print(
                f"      {candidate['name']} | {candidate['periodDays']:.8f} d | "
                f"{candidate['separationArcsec']:.4f}\""
            )
        return {
            "blindName": blind_name,
            "ticID": int(target["ticID"]),
            "frozen": frozen,
            "match": match,
            "status": "REVEAL MATCH FAILED",
        }

    source = match["source"]
    published_period = float(source["periodDays"])

    print()
    print("🔓 AAVSO VSX ANSWER KEY")
    print(f"   VSX name: {source['name']}")
    print(f"   VSX type: {source['type']}")
    print(f"   published period: {published_period:.8f} d")
    print(f"   reveal separation: {source['separationArcsec']:.4f} arcsec")
    print(
        "   separation reproduction error: "
        f"{match['separationDifferenceArcsec']:.4f} arcsec"
    )

    openstar = comparison(
        frozen["openstarPeriodDays"],
        frozen["openstarFrequency"],
        published_period,
    )
    astropy = comparison(
        frozen["astropyPeriodDays"],
        frozen["astropyFrequency"],
        published_period,
    )

    print()
    print("📐 Independent comparison")
    print_comparison("OpenStar", openstar)
    print_comparison("Astropy", astropy)

    return {
        "blindName": blind_name,
        "ticID": int(target["ticID"]),
        "frozen": frozen,
        "vsx": source,
        "match": {
            "expectedSeparationArcsec": match["expectedSeparationArcsec"],
            "revealedSeparationArcsec": source["separationArcsec"],
            "separationDifferenceArcsec": match["separationDifferenceArcsec"],
            "candidateCount": match["candidateCount"],
        },
        "openstarVsPublished": openstar,
        "astropyVsPublished": astropy,
        "status": "REVEALED",
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Reveal the hidden AAVSO VSX periods for the completed "
            "OpenStar blind published-period validation project."
        )
    )
    parser.add_argument("--lock", type=Path, default=LOCK_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    if not args.lock.exists():
        raise RuntimeError(f"Missing preregistration lock: {args.lock}")

    with args.lock.open("r", encoding="utf-8") as file:
        lock_document = json.load(file)

    if lock_document.get("projectID") != PROJECT_ID:
        raise RuntimeError("Lock project ID does not match the completed blind project.")

    stored_sha = lock_document.get("lockSHA256")
    calculated_sha = calculate_lock_sha256(lock_document)

    if stored_sha != calculated_sha:
        raise RuntimeError(
            "Preregistration lock SHA-256 mismatch.\n"
            f"stored:     {stored_sha}\n"
            f"calculated: {calculated_sha}"
        )

    targets = lock_document.get("targets", [])
    if len(targets) != len(FROZEN_RESULTS):
        raise RuntimeError("Target count does not match the frozen completed-run result set.")

    print("🔓 OpenStar Blind Published-Period Validation — REVEAL")
    print(f"project: {PROJECT_ID}")
    print("completed run: 8188 accepted + 4 hard-failed = 8192 terminal work units")
    print("frozen OpenStar/Astropy answers loaded: YES")
    print("VSX Period is being requested NOW.")
    print(f"lock SHA-256 verified: {stored_sha}")

    revealed = []
    for target in targets:
        revealed.append(reveal_target(target))

    output = {
        "projectID": PROJECT_ID,
        "selectionLockSHA256": stored_sha,
        "revealVersion": 1,
        "externalAnswerKey": "AAVSO VSX Period",
        "completedRun": {
            "acceptedWorkUnits": 8188,
            "hardFailedWorkUnits": 4,
            "terminalWorkUnits": 8192,
        },
        "targets": revealed,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=2, allow_nan=False)

    print()
    print()
    print("🏁 REVEAL SUMMARY")
    print("════════════════════════════════════════════════════════")

    for item in revealed:
        if item["status"] != "REVEALED":
            print(f"{item['blindName']}: {item['status']}")
            continue

        published = item["vsx"]["periodDays"]
        openstar_error = item["openstarVsPublished"]["periodErrorPercent"]
        astropy_error = item["astropyVsPublished"]["periodErrorPercent"]

        print(
            f"{item['blindName']}: VSX {published:.8f} d | "
            f"OpenStar Δ {openstar_error:.4f}% | "
            f"Astropy Δ {astropy_error:.4f}%"
        )

    print()
    print(f"💾 Reveal record: {args.output}")


if __name__ == "__main__":
    main()
