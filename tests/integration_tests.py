# tests/integration_tests.py
import subprocess
import sys
from pathlib import Path

import pytest


def run_cmd(args, cwd=None):
    proc = subprocess.run(
        args,
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    return proc.returncode, proc.stdout, proc.stderr


@pytest.fixture
def temp_templates_csv(tmp_path: Path) -> Path:
    """
    Create a minimal templates CSV for integration tests so we don't depend on repo games.csv contents.
    """
    csv_text = """canonical,aliases,l1_B,l1_P,l1_R,l1_H,l1_C,l1_G,l1_rateup_desired,l2_B,l2_P,l2_R,l2_H,l2_C,l2_G,l2_rateup_desired,notes
Example Split,SplitExample|EXS,0.01,,,100,0.5,2,true,0.05,,,10,0.5,2,true,"test template"
Example Single,SingleExample|EX1,0.02,,,,0.5,1,true,,,,,,,,"single only"
"""
    p = tmp_path / "games.csv"
    p.write_text(csv_text, encoding="utf-8")
    return p


def banner_py_path() -> str:
    # Repo root execution: python banner.py ...
    return str(Path("banner.py"))


def test_cli_rate_manual_ok(temp_templates_csv):
    # manual mode: no template, provide required params
    code, out, err = run_cmd([
        sys.executable, banner_py_path(),
        "--templates-csv", str(temp_templates_csv),
        "rate",
        "--B", "0.2",
        "--C", "0.0",
        "--W", "1.0",
    ])
    assert code == 0, err
    assert "EV" in out or "Expected" in out


def test_cli_rate_manual_missing_required_fails(temp_templates_csv):
    code, out, err = run_cmd([
        sys.executable, banner_py_path(),
        "--templates-csv", str(temp_templates_csv),
        "rate",
        "--B", "0.2",
        # missing --C
    ])
    assert code != 0
    assert ("--C" in err) or ("provide" in err.lower()) or ("manual mode" in err.lower())


def test_cli_rate_template_ok(temp_templates_csv):
    # template positional then subcommand
    code, out, err = run_cmd([
        sys.executable, banner_py_path(),
        "--templates-csv", str(temp_templates_csv),
        "Example Single",
        "rate",
        "--W", "0.5",
    ])
    assert code == 0, err
    assert "EV" in out or "Expected" in out


def test_cli_rate_template_override_param(temp_templates_csv):
    # Override template B to make EV differ; ensure command runs successfully
    code, out, err = run_cmd([
        sys.executable, banner_py_path(),
        "--templates-csv", str(temp_templates_csv),
        "Example Single",
        "rate",
        "--W", "1.0",
        "--B", "0.5",
    ])
    assert code == 0, err
    assert "EV" in out or "Expected" in out


def test_cli_split_rate_template_ok(temp_templates_csv):
    code, out, err = run_cmd([
        sys.executable, banner_py_path(),
        "--templates-csv", str(temp_templates_csv),
        "Example Split",
        "split-rate",
        "--l1_W", "0.0",
        "--l2_W", "1.0",
    ])
    assert code == 0, err
    # Expect three outputs
    assert ("L1" in out) or ("l1" in out.lower())
    assert ("L2" in out) or ("l2" in out.lower())
    assert ("any" in out.lower())


def test_cli_split_rate_template_missing_l2_fails(temp_templates_csv):
    # Example Single has no L2; split-rate should fail
    code, out, err = run_cmd([
        sys.executable, banner_py_path(),
        "--templates-csv", str(temp_templates_csv),
        "Example Single",
        "split-rate",
        "--l1_W", "0.0",
        "--l2_W", "1.0",
    ])
    assert code != 0
    assert ("no l2" in err.lower()) or ("l2" in err.lower())


def test_cli_list_templates(temp_templates_csv):
    code, out, err = run_cmd([
        sys.executable, banner_py_path(),
        "--templates-csv", str(temp_templates_csv),
        "--list-templates",
        "rate",
        "--B", "0.2",
        "--C", "0.0",
        "--W", "1.0",
    ])
    assert code == 0, err
    assert "Example Split" in out
    assert "Example Single" in out


def test_cli_unknown_template_fails(temp_templates_csv):
    code, out, err = run_cmd([
        sys.executable, banner_py_path(),
        "DoesNotExist",
        "--templates-csv", str(temp_templates_csv),
        "rate",
        "--W", "0.2",
    ])
    assert code != 0
    assert ("unknown" in err.lower()) or ("keyerror" in err.lower()) or ("template" in err.lower())
