.PHONY: test test-trace test-cov test-full open-trace open-cov

# Run tests (no coverage, no tracing)
test:
	.venv.nosync/bin/python -m pytest

# Per-test traceability matrix (zero external deps)
test-trace:
	.venv.nosync/bin/python -m pytest --traceability

# Aggregate coverage (requires: pip install pytest-cov coverage[toml])
test-cov:
	.venv.nosync/bin/python -m pytest --cov --cov-report=html --cov-report=term-missing

# Both aggregate coverage + per-test traceability
test-full:
	.venv.nosync/bin/python -m pytest --cov --cov-report=html --cov-report=term-missing --traceability

# Open traceability markdown report
open-trace:
	open coverage_trace/traceability.md

# Open HTML coverage report in browser (requires pytest-cov)
open-cov:
	open coverage_html/index.html
