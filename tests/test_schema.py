import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator


@pytest.fixture
def validator():
    schema_path = Path("results/schema.json")
    with schema_path.open() as f:
        schema = json.load(f)
    return Draft202012Validator(schema)


@pytest.fixture
def valid_result():
    return {
        "bench": "unit-test",
        "class": "qwen3-8b-dflash",
        "concurrency": 4,
        "speculation": True,
        "K": 8,
        "tau": 2.5,
        "alpha": 0.7,
        "per_pos": [0.8, 0.6],
        "output_tokens_per_second": 100.0,
        "ttft_ms": 120.0,
        "tpot_ms": 25.0,
        "goodput_slo": 0.8,
    }


def test_valid_result_passes(validator, valid_result):
    assert validator.is_valid(valid_result)


def test_alpha_above_one_fails(validator, valid_result):
    valid_result["alpha"] = 1.2
    assert not validator.is_valid(valid_result)


def test_missing_required_field_fails(validator, valid_result):
    del valid_result["ttft_ms"]
    assert not validator.is_valid(valid_result)


def test_unexpected_field_fails(validator, valid_result):
    valid_result["fake_speedup"] = 99
    assert not validator.is_valid(valid_result)


def test_negative_latency_fails(validator, valid_result):
    valid_result["ttft_ms"] = -1
    assert not validator.is_valid(valid_result)
