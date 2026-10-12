from collections.abc import Sequence


def acceptance_rate(accepted_tokens: int, proposed_tokens: int) -> float:
    """Fraction of proposed draft tokens accepted by the verifier."""
    if accepted_tokens < 0 or proposed_tokens < 0:
        raise ValueError("Token counts cannot be negative.")
    if proposed_tokens == 0:
        raise ValueError("Proposed tokens must be greater than zero.")
    if accepted_tokens > proposed_tokens:
        raise ValueError("Accepted tokens cannot exceed proposed tokens.")
    return accepted_tokens / proposed_tokens


def distributional_overlap(
    target_probs: Sequence[float],
    draft_probs: Sequence[float],
) -> float:
    """Calculate alpha = sum(min(p_v, q_v)) for two distributions."""
    if not target_probs or len(target_probs) != len(draft_probs):
        raise ValueError("Probability vectors must have equal, nonzero length.")

    for distribution in (target_probs, draft_probs):
        if any(p < 0 or p > 1 for p in distribution):
            raise ValueError("Probabilities must be between zero and one.")
        if abs(sum(distribution) - 1.0) > 1e-6:
            raise ValueError("Each probability vector must sum to one.")

    return sum(min(p, q) for p, q in zip(target_probs, draft_probs))


def per_position_acceptance(
    accepted_by_position: Sequence[int],
    proposed_by_position: Sequence[int],
) -> list[float | None]:
    """Calculate acceptance fraction independently at each draft position."""
    if len(accepted_by_position) != len(proposed_by_position):
        raise ValueError("Position arrays must have equal length.")

    rates = []
    for accepted, proposed in zip(
        accepted_by_position, proposed_by_position
    ):
        if accepted < 0 or proposed < 0 or accepted > proposed:
            raise ValueError("Invalid accepted/proposed counts.")
        rates.append(accepted / proposed if proposed else None)

    return rates
