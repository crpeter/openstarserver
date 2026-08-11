from __future__ import annotations

from dataclasses import dataclass


ALLOWED_CLAIMS = {
    "SIGNAL_DETECTED",
    "KNOWN_PERIOD_RECOVERED",
    "KNOWN_PHENOMENON_EXPLAINED",
    "CANDIDATE_PERIOD",
    "INDEPENDENT_PERIOD_ESTIMATE",
    "APPARENTLY_UNCATALOGED_CANDIDATE",
    "HIGH_PRIORITY_DISCOVERY_CANDIDATE",
    "HUMAN_REVIEW_REQUIRED",
}


@dataclass(frozen=True)
class ClaimDecision:
    claim: str
    rationale: tuple[str, ...]

    def as_dict(self) -> dict:
        return {
            "claim": self.claim,
            "rationale": list(self.rationale),
        }


def validate_claim(claim: str) -> str:
    value = str(claim).strip().upper()
    if value == "DISCOVERY":
        raise ValueError("OpenStar must never automatically emit DISCOVERY.")
    if value not in ALLOWED_CLAIMS:
        raise ValueError(f"Unsupported OpenStar claim level: {value}")
    return value


def decision(claim: str, *reasons: str) -> ClaimDecision:
    return ClaimDecision(
        claim=validate_claim(claim),
        rationale=tuple(str(reason) for reason in reasons if str(reason).strip()),
    )
