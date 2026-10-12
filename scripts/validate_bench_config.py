#!/usr/bin/env python3
"""Validate DFlash benchmark configurations before running experiments."""

import argparse
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs"


def load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path.name} must contain a YAML mapping.")
    return data


def validate_configs(config_dir: Path = CONFIG_DIR) -> list[str]:
    errors = []

    try:
        bench = load_yaml(config_dir / "bench_slo.yaml")
        traffic = load_yaml(config_dir / "traffic_classes.yaml")
        serve = load_yaml(config_dir / "serve_dflash_qwen3_8b.yaml")
    except (OSError, yaml.YAMLError, ValueError) as exc:
        return [str(exc)]

    concurrency = bench.get("concurrency")
    if not isinstance(concurrency, list) or not concurrency:
        errors.append("concurrency must be a non-empty list.")
    elif any(type(n) is not int or n < 1 for n in concurrency):
        errors.append("concurrency entries must be positive integers.")
    elif len(concurrency) != len(set(concurrency)):
        errors.append("concurrency entries must be unique.")

    slo = bench.get("slo")
    if not isinstance(slo, dict):
        errors.append("slo must be a mapping.")
    else:
        for key in ("ttft_ms", "tpot_ms"):
            value = slo.get(key)
            if type(value) not in (int, float) or value <= 0:
                errors.append(f"slo.{key} must be a positive number.")

    modes = bench.get("speculation")
    if not isinstance(modes, list) or not modes:
        errors.append("speculation must be a non-empty list.")
    else:
        enabled_values = [
            mode.get("enabled")
            for mode in modes
            if isinstance(mode, dict) and type(mode.get("enabled")) is bool
        ]
        if len(enabled_values) != len(modes):
            errors.append("Each speculation entry must have a boolean enabled field.")
        if True not in enabled_values or False not in enabled_values:
            errors.append("Benchmark matrix must include baseline and speculative modes.")
        for mode in modes:
            if isinstance(mode, dict) and mode.get("enabled") is True:
                k = mode.get("num_speculative_tokens")
                if type(k) is not int or k < 1:
                    errors.append(
                        "Enabled speculation requires positive num_speculative_tokens."
                    )

    classes = traffic.get("classes")
    if not isinstance(classes, list) or not classes:
        errors.append("traffic classes must be a non-empty list.")
    elif any(not isinstance(name, str) or not name.strip() for name in classes):
        errors.append("Traffic class names must be non-empty strings.")
    elif len(classes) != len(set(classes)):
        errors.append("Traffic class names must be unique.")

    target = serve.get("target_model", {})
    speculative = serve.get("speculative", {})
    server = serve.get("server", {})
    generation = serve.get("generation", {})

    if not isinstance(target, dict) or not target.get("name"):
        errors.append("Serving configuration requires target_model.name.")

    if not isinstance(speculative, dict):
        errors.append("speculative must be a mapping.")
    else:
        if speculative.get("method") != "dflash":
            errors.append("speculative.method must be 'dflash'.")
        serving_k = speculative.get("num_speculative_tokens")
        enabled_k = [
            mode.get("num_speculative_tokens")
            for mode in modes or []
            if isinstance(mode, dict) and mode.get("enabled") is True
        ] if isinstance(modes, list) else []
        if enabled_k and serving_k != enabled_k[0]:
            errors.append(
                "Serving num_speculative_tokens must match the benchmark matrix."
            )

    if not isinstance(server, dict):
        errors.append("server must be a mapping.")
    else:
        for key in ("tensor_parallel_size", "max_model_len"):
            value = server.get(key)
            if type(value) is not int or value < 1:
                errors.append(f"server.{key} must be a positive integer.")

    if not isinstance(generation, dict):
        errors.append("generation must be a mapping.")
    else:
        max_tokens = generation.get("max_tokens")
        if type(max_tokens) is not int or max_tokens < 1:
            errors.append("generation.max_tokens must be a positive integer.")
        temperature = generation.get("temperature")
        if type(temperature) not in (int, float) or temperature < 0:
            errors.append("generation.temperature must be non-negative.")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config-dir", type=Path, default=CONFIG_DIR,
        help="Directory containing the three YAML configuration files",
    )
    args = parser.parse_args()

    errors = validate_configs(args.config_dir)
    if errors:
        print("Benchmark configuration: INVALID", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    print("Benchmark configuration: VALID")
    serve = load_yaml(args.config_dir / "serve_dflash_qwen3_8b.yaml")
    model_ref = serve.get("speculative", {}).get("model", "")
    if "<" in model_ref or ">" in model_ref:
        print("Serving readiness: NOT READY (speculator model reference is a placeholder)")
    else:
        print("Serving readiness: placeholder check passed; model availability is unverified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
