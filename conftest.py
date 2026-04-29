"""
Root conftest.py — Per-test code coverage traceability plugin.

Uses Python's built-in sys.settrace() — ZERO external dependencies.
For aggregate coverage, also supports pytest-cov (--cov) if installed.

Usage:
    pytest --traceability          # generate per-test traceability matrix
    pytest --cov --traceability    # aggregate coverage + per-test traceability (needs pytest-cov)

Outputs (when --trace is active):
    coverage_trace/traceability.json   — machine-readable per-test mapping
    coverage_trace/traceability.md     — human-readable markdown summary
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from unittest.mock import MagicMock, patch
import pytest

# ─── Configuration ────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent

# Source directories to track (relative to project root)
SOURCE_DIRS = ["backend", "wellness_env", "rl_training"]

# Additional top-level source files to track
SOURCE_FILES = ["app.py", "inference.py"]

# Absolute paths for fast prefix matching in the trace function
_SOURCE_DIR_PREFIXES = tuple(str(PROJECT_ROOT / d) for d in SOURCE_DIRS)
_SOURCE_FILE_ABSPATHS = set(str(PROJECT_ROOT / f) for f in SOURCE_FILES)

# Patterns to exclude
_EXCLUDE_PREFIXES = (
    str(PROJECT_ROOT / "tests"),
    str(PROJECT_ROOT / ".venv"),
    str(PROJECT_ROOT / "conftest.py"),
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _collapse_to_ranges(lines: list[int]) -> list[str]:
    """Collapse a sorted list of ints into compact range strings.

    Example: [1,2,3,7,8,10] -> ["1-3", "7-8", "10"]
    """
    if not lines:
        return []
    ranges: list[str] = []
    start = prev = lines[0]
    for n in lines[1:]:
        if n == prev + 1:
            prev = n
        else:
            ranges.append(f"{start}-{prev}" if start != prev else str(start))
            start = prev = n
    ranges.append(f"{start}-{prev}" if start != prev else str(start))
    return ranges


def _make_relative(filepath: str) -> str | None:
    """Convert an absolute path to a project-relative path, or None if outside project."""
    try:
        return str(Path(filepath).resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return None


def _is_tracked(filename: str) -> bool:
    """Fast check if a filename (absolute path) is a tracked source file."""
    if filename in _SOURCE_FILE_ABSPATHS:
        return True
    if filename.startswith(_EXCLUDE_PREFIXES):
        return False
    return filename.startswith(_SOURCE_DIR_PREFIXES)


# ─── Trace Function ──────────────────────────────────────────────────────────

class LineTracer:
    """Lightweight line tracer using sys.settrace().

    Collects {abs_filepath: set(line_numbers)} for all tracked source files.
    """

    def __init__(self):
        self.data: dict[str, set[int]] = defaultdict(set)
        self._active = False

    def start(self):
        self.data.clear()
        self._active = True
        sys.settrace(self._trace_calls)
        threading.settrace(self._trace_calls)

    def stop(self):
        self._active = False
        sys.settrace(None)
        threading.settrace(None)

    def _trace_calls(self, frame, event, arg):
        """Trace function for call events — returns line tracer for tracked files."""
        if not self._active:
            return None
        filename = frame.f_code.co_filename
        if _is_tracked(filename):
            self.data[filename].add(frame.f_lineno)
            return self._trace_lines
        return None

    def _trace_lines(self, frame, event, arg):
        """Trace function for line events — records every executed line."""
        if event == "line":
            filename = frame.f_code.co_filename
            self.data[filename].add(frame.f_lineno)
        return self._trace_lines

    def get_results(self) -> dict[str, dict[str, Any]]:
        """Return structured results keyed by relative file paths."""
        files_touched: dict[str, dict[str, Any]] = {}
        for abs_path, lines in self.data.items():
            relpath = _make_relative(abs_path)
            if relpath is None:
                continue
            sorted_lines = sorted(lines)
            if not sorted_lines:
                continue
            files_touched[relpath] = {
                "lines": sorted_lines,
                "line_ranges": _collapse_to_ranges(sorted_lines),
                "total_lines": len(sorted_lines),
            }
        return files_touched


# ─── Plugin State ─────────────────────────────────────────────────────────────

class TraceabilityState:
    """Holds per-test tracing results across the session."""

    def __init__(self):
        self.enabled: bool = False
        self.results: dict[str, dict[str, Any]] = {}
        self._tracer: LineTracer | None = None
        self._current_start: float = 0.0


_state = TraceabilityState()


# ─── Pytest Hooks ─────────────────────────────────────────────────────────────

def pytest_addoption(parser: pytest.Parser):
    """Register the --trace CLI flag."""
    parser.addoption(
        "--traceability",
        action="store_true",
        default=False,
        help="Enable per-test traceability matrix generation.",
    )


def pytest_configure(config: pytest.Config):
    """Activate tracing if --trace was passed."""
    if config.getoption("--traceability", default=False):
        _state.enabled = True


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item: pytest.Item):
    """Wrap only the test call phase with a line tracer.

    Using pytest_runtest_call (not pytest_runtest_protocol) ensures that
    pytest's own makereport hooks fire outside the trace, so pass/fail
    status is correctly captured.
    """
    if not _state.enabled:
        yield
        return

    tracer = LineTracer()
    _state._tracer = tracer
    start = time.monotonic()

    tracer.start()
    yield  # ← the actual test function runs here
    tracer.stop()

    duration = time.monotonic() - start

    try:
        files_touched = tracer.get_results()
        node_id = item.nodeid
        _state.results[node_id] = {
            "status": "executed",  # updated by makereport hook
            "duration_s": round(duration, 4),
            "files_touched": files_touched,
            "total_files": len(files_touched),
            "total_lines": sum(f["total_lines"] for f in files_touched.values()),
        }
    except Exception as exc:
        _state.results[item.nodeid] = {
            "status": "trace_error",
            "error": str(exc),
            "duration_s": round(duration, 4),
            "files_touched": {},
            "total_files": 0,
            "total_lines": 0,
        }
    finally:
        _state._tracer = None


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call):
    """Capture the test outcome and update the traceability record."""
    outcome = yield
    report = outcome.get_result()

    if not _state.enabled:
        return

    if report.when == "call" and item.nodeid in _state.results:
        if report.passed:
            _state.results[item.nodeid]["status"] = "passed"
        elif report.failed:
            _state.results[item.nodeid]["status"] = "failed"
        elif report.skipped:
            _state.results[item.nodeid]["status"] = "skipped"


def pytest_sessionfinish(session: pytest.Session, exitstatus: int):
    """Write traceability outputs at the end of the session."""
    if not _state.enabled or not _state.results:
        return

    out_dir = PROJECT_ROOT / "coverage_trace"
    out_dir.mkdir(exist_ok=True)

    # ── Compute summary stats ──
    total_tests = len(_state.results)
    all_files: set[str] = set()
    total_lines_list: list[int] = []
    total_files_list: list[int] = []

    for r in _state.results.values():
        all_files.update(r.get("files_touched", {}).keys())
        total_lines_list.append(r.get("total_lines", 0))
        total_files_list.append(r.get("total_files", 0))

    summary = {
        "total_tests": total_tests,
        "total_source_files_touched": len(all_files),
        "avg_files_per_test": round(sum(total_files_list) / max(total_tests, 1), 1),
        "avg_lines_per_test": round(sum(total_lines_list) / max(total_tests, 1), 1),
    }

    # ── Build reverse index: file → tests that touch it ──
    reverse_index: dict[str, list[str]] = defaultdict(list)
    for node_id, data in _state.results.items():
        for fpath in data.get("files_touched", {}):
            reverse_index[fpath].append(node_id)

    # ── Compute per-file aggregate coverage ──
    file_coverage: dict[str, dict[str, Any]] = {}
    for filepath in sorted(all_files):
        all_lines_for_file: set[int] = set()
        for data in _state.results.values():
            ft = data.get("files_touched", {})
            if filepath in ft:
                all_lines_for_file.update(ft[filepath]["lines"])

        # Count total executable lines in file
        abs_path = str(PROJECT_ROOT / filepath)
        total_file_lines = 0
        try:
            with open(abs_path) as fh:
                for i, line in enumerate(fh, 1):
                    stripped = line.strip()
                    # Skip blank lines, comments, and docstrings (approximate)
                    if stripped and not stripped.startswith("#") and not stripped.startswith('"""') and not stripped.startswith("'''"):
                        total_file_lines += 1
        except OSError:
            total_file_lines = 0

        covered_lines = len(all_lines_for_file)
        pct = round(100 * covered_lines / max(total_file_lines, 1), 1)
        file_coverage[filepath] = {
            "covered_lines": covered_lines,
            "total_executable_lines": total_file_lines,
            "coverage_pct": pct,
        }

    # ── Write JSON ──
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": summary,
        "file_coverage": file_coverage,
        "tests": _state.results,
        "reverse_index": {k: sorted(v) for k, v in reverse_index.items()},
    }
    json_path = out_dir / "traceability.json"
    with open(json_path, "w") as fh:
        json.dump(payload, fh, indent=2, default=str)

    # ── Write Markdown ──
    md_path = out_dir / "traceability.md"
    with open(md_path, "w") as fh:
        fh.write("# Test ↔ Code Traceability Matrix\n\n")
        fh.write(f"*Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}*\n\n")

        # Summary
        fh.write("## Summary\n\n")
        fh.write("| Metric | Value |\n")
        fh.write("|---|---|\n")
        fh.write(f"| Total tests | {summary['total_tests']} |\n")
        fh.write(f"| Source files touched | {summary['total_source_files_touched']} |\n")
        fh.write(f"| Avg files per test | {summary['avg_files_per_test']} |\n")
        fh.write(f"| Avg lines per test | {summary['avg_lines_per_test']} |\n")
        fh.write("\n")

        # File coverage summary
        fh.write("## File Coverage Summary\n\n")
        fh.write("| File | Covered Lines | Total Lines | Coverage % | Tests |\n")
        fh.write("|---|---|---|---|---|\n")
        for filepath in sorted(file_coverage.keys()):
            fc = file_coverage[filepath]
            n_tests = len(reverse_index.get(filepath, []))
            bar = _coverage_bar(fc["coverage_pct"])
            fh.write(
                f"| `{filepath}` | {fc['covered_lines']} | {fc['total_executable_lines']} "
                f"| {bar} {fc['coverage_pct']}% | {n_tests} |\n"
            )
        fh.write("\n")

        # Per-test table
        fh.write("## Per-Test Traceability\n\n")
        fh.write("| Test Case | Status | Files | Lines | Key Modules |\n")
        fh.write("|---|---|---|---|---|\n")
        for node_id, data in sorted(_state.results.items()):
            status = data.get("status", "?")
            status_icon = {"passed": "✅", "failed": "❌", "skipped": "⏭️"}.get(status, "❓")
            n_files = data.get("total_files", 0)
            n_lines = data.get("total_lines", 0)
            touched = data.get("files_touched", {})
            top_files = sorted(touched.items(), key=lambda x: x[1]["total_lines"], reverse=True)[:3]
            key_modules = ", ".join(Path(f).name for f, _ in top_files)
            fh.write(f"| `{node_id}` | {status_icon} {status} | {n_files} | {n_lines} | {key_modules} |\n")
        fh.write("\n")

        # Reverse index: file → tests
        fh.write("## Reverse Index: File → Tests\n\n")
        fh.write("Which tests exercise each source file.\n\n")
        for filepath in sorted(reverse_index.keys()):
            tests = reverse_index[filepath]
            fc = file_coverage.get(filepath, {})
            pct = fc.get("coverage_pct", 0)
            fh.write(f"### `{filepath}` — {pct}% coverage ({len(tests)} tests)\n\n")
            for t in sorted(tests):
                status = _state.results[t].get("status", "?")
                status_icon = {"passed": "✅", "failed": "❌", "skipped": "⏭️"}.get(status, "❓")
                line_info = _state.results[t]["files_touched"][filepath]
                ranges = ", ".join(line_info["line_ranges"][:10])
                if len(line_info["line_ranges"]) > 10:
                    ranges += f" … (+{len(line_info['line_ranges']) - 10} more)"
                fh.write(f"- {status_icon} `{t}` — lines: {ranges}\n")
            fh.write("\n")

    # ── Terminal output ──
    print(f"\n{'='*70}")
    print(f"  TRACEABILITY: {total_tests} tests traced → {len(all_files)} source files")
    print(f"  JSON  : {json_path}")
    print(f"  Report: {md_path}")
    print(f"{'='*70}")

    # Quick file coverage summary to terminal
    print(f"\n  {'File':<45} {'Coverage':>10}")
    print(f"  {'─'*45} {'─'*10}")
    for filepath in sorted(file_coverage.keys()):
        fc = file_coverage[filepath]
        pct = fc["coverage_pct"]
        bar = _coverage_bar(pct)
        print(f"  {filepath:<45} {bar} {pct:>5.1f}%")
    print()


def _coverage_bar(pct: float) -> str:
    """Return a colored emoji bar for coverage percentage."""
    if pct >= 80:
        return "🟢"
    elif pct >= 50:
        return "🟡"
    elif pct > 0:
        return "🟠"
    else:
        return "🔴"

@pytest.fixture(autouse=True)
def mock_openai():
    """Mock OpenAI client only if API key is missing."""
    if os.environ.get("OPENAI_API_KEY"):
        yield None
        return

    with patch("openai.OpenAI") as mock:
        instance = mock.return_value
        # Configure a default mock response for chat.completions.create
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content='{"calories": 100, "protein_g": 5}'))
        ]
        instance.chat.completions.create.return_value = mock_response
        yield mock
