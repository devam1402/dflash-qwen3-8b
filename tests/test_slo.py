import pytest

from eval.slo import goodput_fraction, request_meets_slo


def test_request_meets_both_slos():
    assert request_meets_slo(100, 20, 200, 30)


def test_request_fails_if_ttft_exceeds_limit():
    assert not request_meets_slo(250, 20, 200, 30)


def test_request_fails_if_tpot_exceeds_limit():
    assert not request_meets_slo(100, 40, 200, 30)


def test_goodput_fraction():
    result = goodput_fraction(
        ttft_ms=[100, 250, 150, 100],
        tpot_ms=[20, 20, 40, 25],
        max_ttft_ms=200,
        max_tpot_ms=30,
    )
    assert result == pytest.approx(0.5)


def test_goodput_rejects_empty_requests():
    with pytest.raises(ValueError):
        goodput_fraction([], [], 200, 30)
