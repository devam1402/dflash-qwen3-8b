import pytest

from eval.metrics import (
    acceptance_rate,
    distributional_overlap,
    per_position_acceptance,
)


def test_acceptance_rate():
    assert acceptance_rate(6, 10) == pytest.approx(0.6)


def test_acceptance_rate_rejects_invalid_counts():
    with pytest.raises(ValueError):
        acceptance_rate(11, 10)


def test_distributional_overlap():
    assert distributional_overlap(
        [0.7, 0.3], [0.4, 0.6]
    ) == pytest.approx(0.7)


def test_per_position_acceptance():
    assert per_position_acceptance(
        [8, 5, 0], [10, 10, 0]
    ) == [0.8, 0.5, None]
