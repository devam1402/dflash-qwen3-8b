#!/usr/bin/env python3
"""Validate DFlash benchmark JSON and generate a Markdown report."""

import argparse
import json
import sys
from pathlib import Path

from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCHEMA = ROOT / "results" / "schema.json"


def load_result(path: Path, schema_path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        result = json.load(f)

    with schema_path.open(encoding="utf-8") as f:
        schema = json.load(f)

    Draft202012Validator.check_schema(schema)
    errors = sorted(
        Draft202012Validator(schema).iter_errors(result),
        key=lambda error: list(map(str, error.absolute_path)),
    )

    if errors:
        details = "\n".join(
            f"- {'.'.join(map(str, error.absolute_path)) or '<root>'}: "
            f"{error.message}"
            for error in errors
        )
        raise ValueError(f"Benchmark result failed schema validation:\n{details}")

    return result


def display(value, suffix=""):
    if value is None:
        return "Not measured"
    return f"{value}{suffix}"


def render_report(result: dict, source: Path) -> str:
    """Render validated metrics and available provenance metadata."""
    metadata = result.get("metadata", {})
    run_type = metadata.get("run_type", "unspecified")

    lines = [
        "# DFlash Benchmark Report",
        "",
        f"> Source: `{source}`",
        f"> Run type: **{run_type}**",
        "> Schema validation checks structure, not whether measurements are genuine.",
        "",
        "## Run details",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Benchmark | {result['bench']} |",
        f"| Workload class | {result['class']} |",
        f"| Concurrency | {result['concurrency']} |",
        f"| Speculation enabled | {result['speculation']} |",
        f"| Configured K | {result['K']} |",
        "",
        "## Metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Mean accepted tokens per verification step (tau) | {display(result['tau'])} |",
        f"| Distributional overlap (alpha) | {display(result['alpha'])} |",
        f"| Per-position acceptance | {', '.join(map(str, result['per_pos'])) or 'Not measured'} |",
        f"| Output throughput (tokens/s) | {result['output_tokens_per_second']} |",
        f"| Time to first token (ms) | {result['ttft_ms']} |",
        f"| Time per output token (ms) | {result['tpot_ms']} |",
        f"| Fraction meeting configured SLOs | {result['goodput_slo']} |",
        "",
        "## Reproducibility metadata",
        "",
    ]

    metadata_labels = [
        ("hardware", "Hardware"),
        ("gpu_name", "GPU"),
        ("gpu_memory_gb", "GPU memory (GB)"),
        ("driver_version", "Driver version"),
        ("cuda_version", "CUDA version"),
        ("python_version", "Python version"),
        ("torch_version", "PyTorch version"),
        ("vllm_version", "vLLM version"),
        ("speculators_version", "Speculators version"),
        ("target_model", "Target model"),
        ("target_model_revision", "Target model revision"),
        ("speculator_revision", "Speculator revision"),
        ("dataset", "Dataset"),
        ("dataset_revision", "Dataset revision"),
        ("config_sha256", "Config SHA-256"),
        ("git_commit", "Git commit"),
        ("notes", "Notes"),
    ]

    found_metadata = False
    for key, label in metadata_labels:
        if key in metadata:
            lines.append(f"- **{label}:** {metadata[key]}")
            found_metadata = True

    if not found_metadata:
        lines.append("No reproducibility metadata supplied.")

    lines.extend([
        "",
        "## Interpretation notes",
        "",
        "- `tau` depends on the documented acceptance convention.",
        "- `alpha` is interpreted as distributional overlap.",
        "- `goodput_slo` is a fraction between 0 and 1, not a percentage.",
        "- Synthetic fixtures are for testing only and must not be presented as measured performance.",
        "- SLO thresholds and detailed workload parameters are not yet recorded in this schema.",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Benchmark result JSON file")
    parser.add_argument(
        "--schema", type=Path, default=DEFAULT_SCHEMA,
        help="JSON schema path",
    )
    parser.add_argument(
        "--output", type=Path,
        help="Markdown output path (default: stdout)",
    )
    args = parser.parse_args()

    try:
        result = load_result(args.input, args.schema)
        report = render_report(result, args.input)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
        print(f"Report written to {args.output}")
    else:
        print(report)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
