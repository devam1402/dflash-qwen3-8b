.PHONY: check

check:
	@echo "=== DFlash Qwen3-8B repository check ==="
	@python3 -c "import yaml; print('PyYAML: OK')"
	@python3 -c "import jsonschema; print('jsonschema: OK')"
	@python3 -c "import yaml, glob; [yaml.safe_load(open(f)) for f in glob.glob('configs/*.yaml')]; print('YAML configs: OK')"
	@python3 -c "import json; json.load(open('results/schema.json')); print('JSON schema: OK')"
	@python3 scripts/validate_bench_config.py
	@python3 -m pytest -q
	@echo "Repository checks passed."
