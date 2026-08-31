"""Runtime for LLM-generated validation experiments.

Generated experiments run in an isolated subprocess. The subprocess receives a
sandbox data directory containing ONLY training rows and the public validation
window. The original benchmark directory is never exposed to generated code.
"""
from __future__ import annotations
import ast
import csv
import json
import os
import py_compile
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict

FORBIDDEN = {"test.csv", "test_data.csv", "test_log.csv", "test_labels", "test_y", "evaluate_test=True", "splits['test']", 'splits["test"]', "splits.get('test')", 'splits.get("test")'}


def validate_source(source: str) -> None:
    if len(source) > 50_000:
        raise ValueError("Generated experiment is too large")
    tree = ast.parse(source, filename="experiment.py")
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                if alias.name in {"subprocess", "socket", "requests", "urllib", "shutil"}:
                    raise ValueError(f"Generated experiment imports prohibited module: {alias.name}")
        if isinstance(node, ast.Call):
            fn = node.func
            name = fn.id if isinstance(fn, ast.Name) else fn.attr if isinstance(fn, ast.Attribute) else ""
            if name in {"system", "popen"}:
                raise ValueError(f"Generated experiment uses prohibited call: {name}")
    lowered = source.lower()
    for marker in FORBIDDEN:
        if marker.lower() in lowered:
            raise ValueError(f"Generated experiment appears to access hidden test data: {marker}")
    if not any(isinstance(n, ast.FunctionDef) and n.name == "run" for n in tree.body):
        raise ValueError("Generated experiment must define run(train_csv, valid_csv, data_dir)")


def _filter_csv(src: str, dst: str, lo: int, hi: int) -> None:
    with open(src, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = [r for r in reader if lo <= int(r["date"]) <= hi]
        fields = reader.fieldnames
    if not fields:
        raise ValueError(f"No CSV header found: {src}")
    with open(dst, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _make_safe_data_dir(data_dir: str, workspace: str) -> str:
    safe = Path(workspace) / "data"
    safe.mkdir()
    shutil.copy2(os.path.join(data_dir, "video_features_basic_pure.csv"), safe / "video_features_basic_pure.csv")
    # data.load() expects these exact filenames. The second file contains ONLY
    # public validation dates, so its 'test' split is empty in the sandbox.
    _filter_csv(os.path.join(data_dir, "log_standard_4_08_to_4_21_pure.csv"),
                str(safe / "log_standard_4_08_to_4_21_pure.csv"), 20220408, 20220421)
    _filter_csv(os.path.join(data_dir, "log_standard_4_22_to_5_08_pure.csv"),
                str(safe / "log_standard_4_22_to_5_08_pure.csv"), 20220422, 20220428)
    return str(safe)


def write_candidate(source: str, workspace: str) -> str:
    validate_source(source)
    path = Path(workspace) / "experiment.py"
    path.write_text(source, encoding="utf-8")
    py_compile.compile(str(path), doraise=True)
    return str(path)


def run_candidate(source: str, repo_dir: str, data_dir: str, timeout: int = 900) -> Dict[str, Any]:
    """Validate and execute one generated experiment with no hidden-test access."""
    with tempfile.TemporaryDirectory(prefix="kuai_generated_") as workspace:
        write_candidate(source, workspace)
        safe_data = _make_safe_data_dir(data_dir, workspace)
        env = os.environ.copy()
        env.update({
            "KUAI_AGENT_MODE": "validation_only",
            "KUAI_TRAIN_CSV": os.path.join(safe_data, "log_standard_4_08_to_4_21_pure.csv"),
            "KUAI_VALID_CSV": os.path.join(safe_data, "log_standard_4_22_to_5_08_pure.csv"),
            "KUAI_DATA_DIR": safe_data,
            "PYTHONPATH": repo_dir + os.pathsep + env.get("PYTHONPATH", ""),
        })
        runner = ("import json, os, experiment; "
                  "r=experiment.run(os.environ['KUAI_TRAIN_CSV'], os.environ['KUAI_VALID_CSV'], os.environ['KUAI_DATA_DIR']); "
                  "print('KUAI_RESULT='+json.dumps(r))")
        proc = subprocess.run([sys.executable, "-c", runner], cwd=workspace, env=env,
                              capture_output=True, text=True, timeout=timeout)
        if proc.returncode != 0:
            raise RuntimeError(f"Generated experiment failed:\n{proc.stderr[-8000:]}")
        for line in reversed(proc.stdout.splitlines()):
            if line.startswith("KUAI_RESULT="):
                result = json.loads(line.split("=", 1)[1])
                if not isinstance(result, dict) or "valid" not in result:
                    raise ValueError("Generated experiment returned no valid metrics")
                return result
        raise ValueError("Generated experiment did not print KUAI_RESULT JSON")
