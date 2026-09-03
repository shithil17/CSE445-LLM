"""tests/test_benchmark.py - validates benchmark_runner.py Markdown table (3×2 = 6 rows, CV mean/std)."""

import re
from pathlib import Path

import pytest

import benchmark_runner


# Helpers
VALID_TABLE = """| Dataset | Algorithm | CV Mean Accuracy | CV Std | Test Accuracy |
|---|---|---|---|---|
| iris | decision_tree | 0.9533 | 0.0340 | 0.9333 |
| iris | random_forest | 0.9667 | 0.0211 | 0.9000 |
| iris | pytorch_mlp | 0.9200 | 0.0200 | 0.9200 |
| breast_cancer | decision_tree | 0.9209 | 0.0202 | 0.9386 |
| breast_cancer | random_forest | 0.9543 | 0.0244 | 0.9561 |
| breast_cancer | pytorch_mlp | 0.9600 | 0.0200 | 0.9600 |"""

INVALID_TABLE_TOO_FEW = """| Dataset | Algorithm | CV Mean Accuracy | CV Std | Test Accuracy |
|---|---|---|---|---|
| iris | decision_tree | 0.95 | 0.03 | 0.93 |
| iris | random_forest | 0.96 | 0.02 | 0.90 |"""


def test_extract_markdown_table_valid():
    text = f"intro\n{VALID_TABLE}\nconclusion"
    table = benchmark_runner._extract_markdown_table(text)
    assert table is not None
    assert "| Dataset | Algorithm |" in table
    assert table.count("\n") >= 7  # header + sep + 6 rows at least
    # should contain all 6 data rows
    lines = [l for l in table.splitlines() if l.strip().startswith("|")]
    assert len(lines) >= 8  # header + sep + 6


def test_extract_markdown_table_invalid():
    assert benchmark_runner._extract_markdown_table("") is None
    assert benchmark_runner._extract_markdown_table("no table here") is None
    assert benchmark_runner._extract_markdown_table("just | one pipe") is None
    # table without separator should be invalid (extract returns None)
    no_sep = "| A | B |\n| 0.9 | 0.1 |\n| 0.8 | 0.2 |"
    assert benchmark_runner._extract_markdown_table(no_sep) is None


def test_is_valid_table_true_for_6_rows():
    assert benchmark_runner._is_valid_table(VALID_TABLE) is True


def test_is_valid_table_false_for_wrong_counts():
    assert benchmark_runner._is_valid_table(INVALID_TABLE_TOO_FEW) is False
    assert benchmark_runner._is_valid_table(None) is False
    assert benchmark_runner._is_valid_table("") is False
    # valid count but non-numeric CV mean
    bad_numeric = VALID_TABLE.replace("0.9533", "n/a")
    assert benchmark_runner._is_valid_table(bad_numeric) is False
    # out-of-range value
    bad_range = VALID_TABLE.replace("0.9533", "1.5")
    assert benchmark_runner._is_valid_table(bad_range) is False


def test_fallback_evaluate_returns_6_row_table():
    md = benchmark_runner._fallback_evaluate()
    assert isinstance(md, str)
    table = benchmark_runner._extract_markdown_table(md)
    assert table is not None, f"fallback produced no table: {md[:500]}"
    assert benchmark_runner._is_valid_table(table) is True
    lines = [l.strip() for l in table.strip().splitlines() if l.strip()]
    # find separator
    sep_idx = next(i for i, l in enumerate(lines) if re.search(r"[-:]{3,}", l))
    data_rows = [l for l in lines[sep_idx + 1 :] if "|" in l]
    assert len(data_rows) == 6, f"expected 6 data rows, got {len(data_rows)}"
    # numeric ranges 0-1 for CV mean/std
    for row in data_rows:
        cols = [c.strip() for c in row.strip().strip("|").split("|")]
        assert len(cols) >= 5
        cv_mean = float(cols[2])
        cv_std = float(cols[3])
        test_acc = float(cols[4])
        assert 0.0 <= cv_mean <= 1.0
        assert 0.0 <= cv_std <= 0.5
        assert 0.0 <= test_acc <= 1.0


def test_run_benchmark_persists_report():
    md = benchmark_runner.run_benchmark()
    report_path = Path("report/benchmark_summary.md")
    assert report_path.exists(), "report/benchmark_summary.md not written"
    content = report_path.read_text(encoding="utf-8")
    assert content.strip() == md.strip() or md.strip() in content.strip()
    table = benchmark_runner._extract_markdown_table(content)
    assert table is not None
    assert benchmark_runner._is_valid_table(table) is True
    # header columns check
    header = table.splitlines()[0].lower()
    assert "dataset" in header
    assert "algorithm" in header
    assert "cv mean" in header or "cv mean accuracy" in header
    assert "cv std" in header
    assert "test accuracy" in header


def test_benchmark_table_content_covers_datasets_and_algos():
    # Use fallback directly for deterministic check; also check persisted file
    md = benchmark_runner._fallback_evaluate()
    table = benchmark_runner._extract_markdown_table(md)
    assert table is not None
    lowered = table.lower()
    assert "iris" in lowered
    assert "breast_cancer" in lowered
    assert "decision_tree" in lowered
    assert "random_forest" in lowered
    assert "pytorch_mlp" in lowered
    # also verify persisted report if exists
    report_path = Path("report/benchmark_summary.md")
    if report_path.exists():
        rt_table = benchmark_runner._extract_markdown_table(report_path.read_text(encoding="utf-8"))
        if rt_table:
            rl = rt_table.lower()
            assert "iris" in rl and "breast_cancer" in rl
            assert "decision_tree" in rl and "random_forest" in rl and "pytorch_mlp" in rl


def test_benchmark_prompt_constants():
    assert hasattr(benchmark_runner, "BENCHMARK_PROMPT")
    assert hasattr(benchmark_runner, "TASK3_PROMPT")
    assert benchmark_runner.BENCHMARK_PROMPT == benchmark_runner.TASK3_PROMPT
    prompt = benchmark_runner.BENCHMARK_PROMPT.lower()
    assert "3 algorithms" in prompt or "3 algorithm" in prompt
    assert "2 datasets" in prompt or "2 dataset" in prompt
    assert "cross-validation" in prompt or "cross validation" in prompt
    assert "markdown" in prompt
