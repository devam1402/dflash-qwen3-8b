import shutil
from pathlib import Path

import yaml

from scripts.validate_bench_config import validate_configs


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs"


def copied_configs(tmp_path):
    destination = tmp_path / "configs"
    shutil.copytree(CONFIG_DIR, destination)
    return destination


def edit_yaml(directory, filename, edit):
    path = directory / filename
    data = yaml.safe_load(path.read_text())
    edit(data)
    path.write_text(yaml.safe_dump(data, sort_keys=False))


def test_current_configs_are_valid():
    assert validate_configs(CONFIG_DIR) == []


def test_duplicate_concurrency_is_rejected(tmp_path):
    directory = copied_configs(tmp_path)
    edit_yaml(
        directory,
        "bench_slo.yaml",
        lambda data: data["concurrency"].append(4),
    )
    assert any("unique" in error for error in validate_configs(directory))


def test_nonpositive_ttft_is_rejected(tmp_path):
    directory = copied_configs(tmp_path)
    edit_yaml(
        directory,
        "bench_slo.yaml",
        lambda data: data["slo"].update(ttft_ms=0),
    )
    assert any("slo.ttft_ms" in error for error in validate_configs(directory))


def test_missing_baseline_is_rejected(tmp_path):
    directory = copied_configs(tmp_path)
    edit_yaml(
        directory,
        "bench_slo.yaml",
        lambda data: data.update(
            speculation=[{"enabled": True, "num_speculative_tokens": 7}]
        ),
    )
    assert any("baseline" in error for error in validate_configs(directory))


def test_serving_token_mismatch_is_rejected(tmp_path):
    directory = copied_configs(tmp_path)
    edit_yaml(
        directory,
        "serve_dflash_qwen3_8b.yaml",
        lambda data: data["speculative"].update(num_speculative_tokens=5),
    )
    assert any("must match" in error for error in validate_configs(directory))


def test_duplicate_traffic_class_is_rejected(tmp_path):
    directory = copied_configs(tmp_path)
    edit_yaml(
        directory,
        "traffic_classes.yaml",
        lambda data: data["classes"].append("coding"),
    )
    assert any(
        "Traffic class names must be unique" in error
        for error in validate_configs(directory)
    )
