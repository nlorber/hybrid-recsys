"""Duration proximity scoring for media recommendations."""


def duration_score(delta: float, penalty: float = -1.0) -> float:
    """Score a media item based on how close its duration is to the preferred duration.

    The delta is defined as: requested_duration - media_duration.
    - delta >= 0: media is shorter than requested (no penalty)
    - delta < 0: media is longer than requested (penalty applied)

    Args:
        delta: Difference between requested and actual duration in seconds.
        penalty: Score adjustment for media longer than preferred. Default -1.0.

    Returns:
        Score in range (penalty, 1.0]. Higher is better.
    """
    base = 1.0 / (delta**2 + 1)
    if delta >= 0:
        return base
    return base + penalty
