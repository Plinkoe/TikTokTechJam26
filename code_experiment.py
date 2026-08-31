"""Runtime for LLM-generated validation experiments."""
from __future__ import annotations
import ast
import json
import os
import py_compile
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


def write_candidate(source: str, workspace: str) -> str:
    validate_source(source)
    path = Path(workspace) / "experiment.py"
    path.write_text(source, encoding="utf-8")
    py_compile.compile(str(path), doraise=True)
    return str(path)


def run_candidate(source: str, repo_dir: str, data_dir: str, timeout: int = 900) -> Dict[str, Any]:
    """Validate and execute one generated experiment in an isolated subprocess."""
    with tempfile.TemporaryDirectory(prefix="kuai_generated_") as workspace:
        write_candidate(source, workspace)
        train_csv = os.path.join(data_dir, "log_standard_4_08_to_4_21_pure.csv")
        valid_csv = os.path.join(data_dir, "log_random_4_22_to_5_08_pure.csv")
        env = os.environ.copy()
        env.update({
            "KUAI_AGENT_MODE": "validation_only",
            "KUAI_TRAIN_CSV": train_csv,
            "KUAI_VALID_CSV": valid_csv,
            "KUAI_DATA_DIR": data_dir,
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
