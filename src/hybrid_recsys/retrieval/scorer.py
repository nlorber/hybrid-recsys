"""Duration proximity scoring for media recommendations."""

_SECONDS_PER_MINUTE: float = 60.0


def duration_score(delta: float, penalty: float = -1.0) -> float:
    """Score a media item based on how close its duration is to the preferred duration.

    The delta is defined as: requested_duration - media_duration (in seconds).
    - delta >= 0: media is shorter than requested (no penalty)
    - delta < 0: media is longer than requested (penalty applied)

    Delta is normalised to minutes before applying the proximity formula so that
    the score remains meaningful for typical podcast durations (hundreds of seconds).
    A one-minute difference yields base ≈ 0.5; ten minutes yields base ≈ 0.01.

    Args:
        delta: Difference between requested and actual duration in seconds.
        penalty: Score adjustment for media longer than preferred. Default -1.0.

    Returns:
        Score in range (penalty, 1.0]. Higher is better.
    """
    delta_min = delta / _SECONDS_PER_MINUTE
    base = 1.0 / (delta_min**2 + 1)
    if delta >= 0:
        return base
    return base + penalty
