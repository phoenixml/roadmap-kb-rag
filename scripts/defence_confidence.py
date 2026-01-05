# scripts/defence_confidence.py

def compute_defence_confidence(
    evidence_count: int,
    applicability: str,
    critic_adjustment: str,
    has_prior: bool,
) -> float:
    """
    Deterministic confidence score for Attack -> Defence edge.
    """

    score = 0.0

    # Evidence strength
    score += min(evidence_count * 0.15, 0.45)

    # Applicability weighting
    score += {
        "high": 0.3,
        "medium": 0.2,
        "low": 0.1,
    }.get(applicability, 0.15)

    # Prior Neo4j memory bonus
    if has_prior:
        score += 0.15

    # Critic signal
    if critic_adjustment == "increase":
        score += 0.1
    elif critic_adjustment == "decrease":
        score -= 0.2

    return round(max(0.0, min(score, 1.0)), 3)
