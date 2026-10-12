import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "10_report.py"
SCHEMA = ROOT / "results" / "schema.json"


def sample_result():
    return {
        "bench": "cpu-fixture",
        "class": "test-workload",
        "concurrency": 1,
        "speculation": False,
        "K": 0,
        "tau": None,
        "alpha": None,
        "per_pos": [],
        "output_tokens_per_second": 0,
        "ttft_ms": 0,
        "tpot_ms": 0,
        "goodput_slo": 0,
    }


def run_report(input_path, output_path):
    return subprocess.run(
        [
            sys.executable, str(SCRIPT), str(input_path),
            "--schema", str(SCHEMA), "--output", str(output_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def test_report_generates_markdown(tmp_path):
    source = tmp_path / "fixture.json"
    output = tmp_path / "report.md"
    source.write_text(json.dumps(sample_result()))

    completed = run_report(source, output)

    assert completed.returncode == 0, completed.stderr
    report = output.read_text()
    assert "# DFlash Benchmark Report" in report
    assert "Not measured" in report
    assert "cpu-fixture" in report


def test_report_rejects_invalid_result(tmp_path):
    source = tmp_path / "invalid.json"
    output = tmp_path / "report.md"
    result = sample_result()
    result["alpha"] = 2
    source.write_text(json.dumps(result))

    completed = run_report(source, output)

    assert completed.returncode == 1
    assert "failed schema validation" in completed.stderr
    assert not output.exists()


def test_report_uses_real_newlines(tmp_path):
    source = tmp_path / "fixture.json"
    output = tmp_path / "report.md"
    source.write_text(json.dumps(sample_result()))

    completed = run_report(source, output)

    assert completed.returncode == 0, completed.stderr
    report = output.read_text()
    assert "# DFlash Benchmark Report\n" in report
    assert "\\n## Run details" not in report
