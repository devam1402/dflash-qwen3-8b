from collections.abc import Sequence


def request_meets_slo(
    ttft_ms: float,
    tpot_ms: float,
    max_ttft_ms: float,
    max_tpot_ms: float,
) -> bool:
    """Return whether a request meets both latency SLOs."""
    values = (ttft_ms, tpot_ms, max_ttft_ms, max_tpot_ms)

    if any(value < 0 for value in values):
        raise ValueError("Latency values cannot be negative.")

    if max_ttft_ms == 0 or max_tpot_ms == 0:
        raise ValueError("SLO thresholds must be greater than zero.")

    return ttft_ms <= max_ttft_ms and tpot_ms <= max_tpot_ms


def goodput_fraction(
    ttft_ms: Sequence[float],
    tpot_ms: Sequence[float],
    max_ttft_ms: float,
    max_tpot_ms: float,
) -> float:
    """Fraction of requests meeting both TTFT and TPOT SLOs."""
    if len(ttft_ms) != len(tpot_ms):
        raise ValueError("TTFT and TPOT arrays must have equal length.")

    if not ttft_ms:
        raise ValueError("At least one request is required.")

    compliant = sum(
        request_meets_slo(ttft, tpot, max_ttft_ms, max_tpot_ms)
        for ttft, tpot in zip(ttft_ms, tpot_ms)
    )

    return compliant / len(ttft_ms)
