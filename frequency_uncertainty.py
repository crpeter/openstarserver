"""Generic profile-likelihood uncertainty for a sinusoidal frequency estimate."""

import math


_CHI_SQUARE_95_ONE_PARAMETER = 3.841458820694124


def _solve_3x3(matrix, vector):
    rows = [list(matrix[index]) + [vector[index]] for index in range(3)]
    for column in range(3):
        pivot = max(range(column, 3), key=lambda row: abs(rows[row][column]))
        if abs(rows[pivot][column]) < 1e-14:
            return None
        rows[column], rows[pivot] = rows[pivot], rows[column]
        scale = rows[column][column]
        rows[column] = [value / scale for value in rows[column]]
        for row in range(3):
            if row == column:
                continue
            scale = rows[row][column]
            rows[row] = [
                rows[row][item] - scale * rows[column][item]
                for item in range(4)
            ]
    return [rows[index][3] for index in range(3)]


def _sinusoid_rss(times, values, frequency, weights):
    columns = []
    for time in times:
        angle = 2.0 * math.pi * frequency * time
        columns.append((1.0, math.cos(angle), math.sin(angle)))
    normal = [[0.0] * 3 for _ in range(3)]
    rhs = [0.0] * 3
    for row, value, weight in zip(columns, values, weights):
        for left in range(3):
            rhs[left] += weight * row[left] * value
            for right in range(3):
                normal[left][right] += weight * row[left] * row[right]
    coefficients = _solve_3x3(normal, rhs)
    if coefficients is None:
        return None
    return sum(
        weight * (value - sum(a * b for a, b in zip(coefficients, row))) ** 2
        for row, value, weight in zip(columns, values, weights)
    )


def estimate_frequency_interval(
    dataset,
    selected_frequency,
    competing_frequencies=(),
    competing_mode_coverage=None,
):
    """Return a 95% local profile-likelihood interval and reliability diagnostics.

    The model is offset + sine + cosine.  With no supplied standard errors the
    residual variance is profiled out.  The routine deliberately returns no
    interval when a boundary truncates it or a competing mode belongs to the
    same 95% likelihood region.
    """
    times = dataset.get("times") or []
    values = dataset.get("flux") or dataset.get("values") or []
    errors = (
        dataset.get("measurementUncertainties")
        or dataset.get("fluxUncertainties")
        or dataset.get("valueUncertainties")
    )
    uncertainties_are_relative = bool(
        dataset.get("measurementUncertaintiesAreRelative", False)
    )
    known_sigma = errors is not None and not uncertainties_are_relative
    diagnostics = {
        "method": "sinusoid-profile-likelihood",
        "confidenceLevel": 0.95,
        "sampleCount": min(len(times), len(values)),
        "measurementUncertaintiesUsed": errors is not None,
        "noiseScaleTreatment": (
            "known-per-observation-standard-deviations"
            if known_sigma
            else "profiled-global-residual-scale"
        ),
        "assumptions": "independent Gaussian residuals; offset plus one sinusoid",
        "trustworthy": False,
    }
    try:
        observations = [
            (float(t), float(y), index)
            for index, (t, y) in enumerate(zip(times, values))
            if math.isfinite(float(t)) and math.isfinite(float(y))
        ]
        frequency = float(selected_frequency)
    except (TypeError, ValueError):
        diagnostics["unavailableReason"] = "non-finite observations or frequency"
        return None, diagnostics
    if len(observations) < 8 or not math.isfinite(frequency) or frequency <= 0:
        diagnostics["unavailableReason"] = "at least eight finite samples are required"
        return None, diagnostics
    clean_times = [item[0] for item in observations]
    clean_values = [item[1] for item in observations]
    baseline = max(clean_times) - min(clean_times)
    diagnostics["baseline"] = baseline
    if baseline <= 0:
        diagnostics["unavailableReason"] = "positive time baseline is required"
        return None, diagnostics
    weights = [1.0] * len(observations)
    if errors is not None:
        try:
            weights = [1.0 / float(errors[item[2]]) ** 2 for item in observations]
            if any(not math.isfinite(weight) or weight <= 0 for weight in weights):
                raise ValueError
        except (IndexError, TypeError, ValueError, ZeroDivisionError):
            diagnostics["unavailableReason"] = "measurement uncertainties are invalid"
            return None, diagnostics

    profile_evaluations = 0

    def evaluate_rss(trial_frequency):
        nonlocal profile_evaluations
        profile_evaluations += 1
        return _sinusoid_rss(
            clean_times, clean_values, trial_frequency, weights
        )

    search = dataset.get("frequencySearch") or {}
    minimum = float(search.get("minimumFrequency", search.get("minFrequency", 0.0)))
    step = float(search.get("frequencyStep", search.get("step", 0.0)))
    count = int(search.get("totalFrequencies", search.get("frequencyCount", 0)))
    maximum = float(search.get("maximumFrequency", minimum + max(count - 1, 0) * step))
    if count <= 0 and step > 0 and maximum >= minimum:
        count = int(math.floor((maximum - minimum) / step)) + 1
    if not (0 < minimum < maximum and minimum <= frequency <= maximum):
        diagnostics["unavailableReason"] = "selected frequency is outside a valid search range"
        return None, diagnostics

    rayleigh = 1.0 / baseline
    # Remove grid quantisation from the statistical calculation by refining
    # the selected peak continuously inside its local Rayleigh neighbourhood.
    refine_lower = max(minimum, frequency - rayleigh / 2.0)
    refine_upper = min(maximum, frequency + rayleigh / 2.0)
    for _ in range(60):
        left = refine_lower + (refine_upper - refine_lower) / 3.0
        right = refine_upper - (refine_upper - refine_lower) / 3.0
        if evaluate_rss(left) <= evaluate_rss(right):
            refine_upper = right
        else:
            refine_lower = left
    frequency = (refine_lower + refine_upper) / 2.0
    rss_best = evaluate_rss(frequency)
    if rss_best is None or rss_best <= 0 or not math.isfinite(rss_best):
        diagnostics["unavailableReason"] = "sinusoid fit is singular or has zero residual variance"
        return None, diagnostics
    sample_count = len(clean_times)
    if known_sigma:
        threshold_rss = rss_best + _CHI_SQUARE_95_ONE_PARAMETER
    else:
        threshold_rss = rss_best * math.exp(
            _CHI_SQUARE_95_ONE_PARAMETER / sample_count
        )
    diagnostics.update({
        "rayleighResolution": rayleigh,
        "profileMaximumFrequency": frequency,
        "selectedFitRSS": rss_best,
    })

    # Find each crossing by expanding away from the selected local optimum,
    # then bisection. This measures the fitted peak rather than grid spacing.
    def crossing(direction):
        inside = frequency
        increment = max(rayleigh / 64.0, step if step > 0 else 0.0)
        outside = inside
        for _ in range(20):
            outside = min(max(frequency + direction * increment, minimum), maximum)
            rss = evaluate_rss(outside)
            if rss is None or rss >= threshold_rss:
                break
            if outside in (minimum, maximum):
                return None
            inside = outside
            increment *= 1.7
        else:
            return None
        if outside in (minimum, maximum) and (rss is not None and rss < threshold_rss):
            return None
        lo, hi = sorted((inside, outside))
        for _ in range(55):
            mid = (lo + hi) / 2.0
            mid_rss = evaluate_rss(mid)
            if (mid_rss < threshold_rss) == (direction > 0):
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2.0

    lower, upper = crossing(-1), crossing(1)
    if lower is None or upper is None:
        diagnostics["unavailableReason"] = "confidence region is truncated by the search boundary"
        return None, diagnostics

    def profile_statistic(rss):
        if known_sigma:
            return rss - rss_best
        return sample_count * math.log(rss / rss_best)

    coverage = competing_mode_coverage or {}
    chunks = coverage.get("chunks") or []
    coverage_complete = bool(coverage.get("complete"))
    objective_matches = bool(coverage.get("objectiveMatches", True))
    selected_worker_power = coverage.get("selectedPower")
    powers_valid = competing_mode_coverage is None or (
        selected_worker_power is not None
        and math.isfinite(float(selected_worker_power))
        and float(selected_worker_power) < 1.0
        and all(
            chunk.get("power") is not None
            and math.isfinite(float(chunk["power"]))
            and float(chunk["power"]) < 1.0
            for chunk in chunks
        )
    )
    maximum_chunk_span = max(
        (abs(float(chunk.get("endFrequency")) - float(chunk.get("startFrequency")))
         for chunk in chunks),
        default=None,
    )
    coverage_sufficient = (
        coverage_complete
        and objective_matches
        and powers_valid
        and maximum_chunk_span is not None
        and maximum_chunk_span <= rayleigh
    )
    diagnostics.update({
        "competingModeSearch": "all-distributed-chunk-maxima",
        "competingModeChunkCount": len(chunks),
        "competingModeCoverageComplete": coverage_complete,
        "competingModeObjectiveMatches": objective_matches,
        "competingModeWorkerPowersValid": powers_valid,
        "maximumChunkFrequencySpan": maximum_chunk_span,
        "competingModeCoverageSufficient": coverage_sufficient,
        "profileModelEvaluations": profile_evaluations,
        "rawChunkCandidatesInspected": 0,
        "competingChunksRefined": 0,
        "powerScreenedChunkCount": 0,
        "boundaryStraddlingChunkCount": 0,
    })
    if competing_mode_coverage is not None and not coverage_sufficient:
        diagnostics["unavailableReason"] = (
            "distributed chunk maxima do not safely cover separated competing modes"
        )
        return None, diagnostics

    plausible_aliases = []
    worker_power_threshold = None
    if competing_mode_coverage is not None:
        selected_worker_power = float(selected_worker_power)
        # Standard Lomb-Scargle power is 1 - RSS/RSS_null for the same
        # offset+sin+cos least-squares objective. Therefore this is exactly
        # the profiled-noise likelihood boundary expressed in worker power.
        worker_power_threshold = 1.0 - (
            (1.0 - selected_worker_power)
            * math.exp(_CHI_SQUARE_95_ONE_PARAMETER / sample_count)
        )
        diagnostics["workerPowerCompetitiveThreshold"] = worker_power_threshold

    for chunk in chunks:
        diagnostics["rawChunkCandidatesInspected"] += 1
        alternative = chunk.get("frequency")
        if alternative is None:
            continue
        alternative = float(alternative)
        if abs(alternative - frequency) < rayleigh:
            if (
                float(chunk["startFrequency"]) < frequency - rayleigh
                or float(chunk["endFrequency"]) > frequency + rayleigh
            ):
                diagnostics["boundaryStraddlingChunkCount"] += 1
                diagnostics["unavailableReason"] = (
                    "a chunk winner inside the local peak has unsearched-for-mode "
                    "coverage outside the Rayleigh exclusion region"
                )
                return None, diagnostics
            continue
        if float(chunk["power"]) < worker_power_threshold:
            diagnostics["powerScreenedChunkCount"] += 1
            continue
        rss = evaluate_rss(alternative)
        if rss is not None and profile_statistic(rss) > _CHI_SQUARE_95_ONE_PARAMETER:
            diagnostics["competingChunksRefined"] += 1
            alternative_lower = max(
                float(chunk["startFrequency"]), alternative - rayleigh / 2.0
            )
            alternative_upper = min(
                float(chunk["endFrequency"]), alternative + rayleigh / 2.0
            )
            for _ in range(30):
                left = alternative_lower + (alternative_upper - alternative_lower) / 3.0
                right = alternative_upper - (alternative_upper - alternative_lower) / 3.0
                if evaluate_rss(left) <= evaluate_rss(right):
                    alternative_upper = right
                else:
                    alternative_lower = left
            alternative = (alternative_lower + alternative_upper) / 2.0
            rss = evaluate_rss(alternative)
        if (
            rss is not None
            and profile_statistic(rss) <= _CHI_SQUARE_95_ONE_PARAMETER
            and not any(abs(alternative - item) <= abs(step) for item in plausible_aliases)
        ):
            plausible_aliases.append(alternative)
    for alternative in competing_frequencies:
        alternative = float(alternative)
        if abs(alternative - frequency) < rayleigh:
            continue
        rss = evaluate_rss(alternative)
        if rss is not None and profile_statistic(rss) <= _CHI_SQUARE_95_ONE_PARAMETER:
            plausible_aliases.append(alternative)
    diagnostics["plausibleCompetingFrequencies"] = plausible_aliases
    diagnostics["intervalWidth"] = upper - lower
    diagnostics["profileModelEvaluations"] = profile_evaluations
    if plausible_aliases:
        diagnostics["unavailableReason"] = "a separated competing peak is inside the 95% likelihood region"
        return None, diagnostics
    diagnostics["trustworthy"] = True
    return {
        "lower": lower,
        "upper": upper,
        "method": "sinusoid-profile-likelihood",
        "confidenceLevel": 0.95,
    }, diagnostics
