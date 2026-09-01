"""Offline regression tests for the LLM planner's failure paths.

These reproduce, with no network calls, the failure that made a forced
code-generation run silently fall back to fm_baseline: a response truncated
at the token cap, which surfaces only as a JSONDecodeError.
"""

import json
import sys
import types
import unittest
from unittest import mock

try:  # llm_planner imports torch transitively, only for a docstring constant.
    import torch  # noqa: F401
except ImportError:  # keep these tests runnable without the ML stack installed
    _stub = types.ModuleType("torch")
    _stub.nn = types.ModuleType("torch.nn")
    _stub.nn.functional = types.ModuleType("torch.nn.functional")
    _stub.nn.Module = object
    _stub.Tensor = object
    _stub.no_grad = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("stub"))
    sys.modules.setdefault("torch", _stub)
    sys.modules.setdefault("torch.nn", _stub.nn)
    sys.modules.setdefault("torch.nn.functional", _stub.nn.functional)

import llm_planner as lp


class _FakeResponse:
    def __init__(self, payload):
        self._data = json.dumps(payload).encode("utf-8")

    def read(self):
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _completion(content, finish_reason="stop", completion_tokens=120):
    return {
        "choices": [{"finish_reason": finish_reason, "message": {"content": content}}],
        "usage": {"prompt_tokens": 500, "completion_tokens": completion_tokens},
    }


def _planner(**overrides):
    cfg = lp.LLMPlannerConfig(
        enabled=True, provider="openrouter", model="test-model",
        api_key="test-key", base_url="https://example.invalid/v1", **overrides
    )
    return lp.LLMPlanner(cfg)


CODE = "class CandidateModel(nn.Module):\n    pass\n"


class CodeExtractionTests(unittest.TestCase):
    def test_code_as_list_of_lines(self):
        proposal = lp.normalize_proposal({
            "proposal": {"mode": "code", "code": ["class CandidateModel(nn.Module):", "    pass"]}
        })
        self.assertEqual(proposal["mode"], "code")
        self.assertIn("class CandidateModel", proposal["code"])
        self.assertIn("\n    pass", proposal["code"])

    def test_code_inside_fenced_block(self):
        proposal = lp.normalize_proposal({
            "proposal": {"mode": "code", "code": "```python\n" + CODE + "```"}
        })
        self.assertTrue(proposal["code"].startswith("class CandidateModel"))
        self.assertNotIn("```", proposal["code"])

    def test_code_mode_defaults_family_to_llm_code(self):
        self.assertEqual(lp.normalize_proposal({"mode": "code", "code": CODE})["family"], "llm_code")


class ProviderFailureTests(unittest.TestCase):
    def test_truncation_is_named_not_disguised_as_bad_json(self):
        planner = _planner(max_tokens=300)
        truncated = '{"proposal": {"mode": "code", "code": "class Candid'
        with mock.patch.object(lp.request, "urlopen",
                               return_value=_FakeResponse(_completion(truncated, "length", 300))):
            with self.assertRaises(lp.TruncatedResponseError) as ctx:
                planner._call_provider({"max_tokens": 300})
        self.assertIn("KUAI_LLM_MAX_TOKENS", str(ctx.exception))

    def test_raw_content_is_captured_for_the_call_log(self):
        planner = _planner()
        records = []
        planner.call_log_hook = records.append
        with mock.patch.object(lp.request, "urlopen",
                               return_value=_FakeResponse(_completion("not json at all"))):
            with self.assertRaises(ValueError):
                planner._call_provider({"max_tokens": 4000})
        self.assertEqual(len(records), 1)
        self.assertFalse(records[0]["success"])
        self.assertEqual(records[0]["raw_content"], "not json at all")
        self.assertEqual(records[0]["finish_reason"], "stop")

    def test_successful_call_is_logged_too(self):
        planner = _planner()
        records = []
        planner.call_log_hook = records.append
        payload = json.dumps({"proposal": {"mode": "code", "code": CODE}})
        with mock.patch.object(lp.request, "urlopen", return_value=_FakeResponse(_completion(payload))):
            planner._call_provider({"max_tokens": 4000})
        self.assertTrue(records[0]["success"])
        self.assertEqual(records[0]["usage"]["completion_tokens"], 120)


class HistorySerializationTests(unittest.TestCase):
    """Validation metrics arrive as numpy scalars; the history summary must
    survive them. This crashed the run the first time a generated candidate
    actually trained and produced real metrics."""

    def test_numpy_scalar_metrics_serialize(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy not installed")
        records = [{
            "experiment": "llm_generated_0", "family": "llm_code", "status": "ok",
            "metrics": {"valid": {"primary": np.float32(0.601), "GAUC": np.float32(0.667),
                                  "nDCG@5": np.float32(0.535), "users": np.int64(22377)}},
        }]
        summary = lp.summarize_for_prompt(records)
        self.assertIn("llm_generated_0", summary)
        self.assertIn("0.60", summary)

    def test_unserializable_object_degrades_to_a_string(self):
        class Weird:
            def __repr__(self):
                return "<weird>"
        summary = lp.summarize_for_prompt([{"experiment": "x", "metrics": {"valid": {"primary": Weird()}}}])
        self.assertIn("weird", summary)


class RetryTests(unittest.TestCase):
    def test_force_code_retries_and_recovers_after_truncation(self):
        planner = _planner(force_code=True, max_retries=3)
        good = json.dumps({"proposal": {"mode": "code", "code": CODE, "hypothesis": "h"}})
        responses = [
            _FakeResponse(_completion('{"proposal": {"mode": "code", "code": "cla', "length", 300)),
            _FakeResponse(_completion(good)),
        ]
        with mock.patch.object(lp.request, "urlopen", side_effect=responses) as urlopen:
            spec = planner.suggest_next_experiment(None, [])
        self.assertIsNotNone(spec)
        self.assertEqual(spec.mode, "code")
        self.assertEqual(urlopen.call_count, 2)

    def test_retry_feeds_the_error_back_and_raises_temperature(self):
        planner = _planner(force_code=True, max_retries=2, temperature=0.2)
        sent = []

        def _capture(req, **kwargs):
            sent.append(json.loads(req.data.decode("utf-8")))
            return _FakeResponse(_completion("garbage"))

        with mock.patch.object(lp.request, "urlopen", side_effect=_capture):
            self.assertIsNone(planner.suggest_next_experiment(None, []))
        self.assertEqual(len(sent), 2)
        self.assertNotIn("PREVIOUS ATTEMPT FAILED", sent[0]["messages"][1]["content"])
        self.assertIn("PREVIOUS ATTEMPT FAILED", sent[1]["messages"][1]["content"])
        self.assertGreater(sent[1]["temperature"], sent[0]["temperature"])

    def test_tune_response_under_force_code_is_rejected_and_retried(self):
        planner = _planner(force_code=True, max_retries=2)
        tune = json.dumps({"proposal": {"mode": "tune", "name": "fm_baseline"}})
        with mock.patch.object(lp.request, "urlopen",
                               side_effect=[_FakeResponse(_completion(tune)),
                                            _FakeResponse(_completion(tune))]) as urlopen:
            self.assertIsNone(planner.suggest_next_experiment(None, []))
        self.assertEqual(urlopen.call_count, 2)
        self.assertIn("FORCE_CODE", planner.last_error)

    def test_retries_apply_without_force_code_too(self):
        # A candidate rejected by the dry-run validator deserves another turn
        # in every mode, not only under FORCE_CODE.
        planner = _planner(force_code=False, max_retries=3)
        with mock.patch.object(lp.request, "urlopen",
                               return_value=_FakeResponse(_completion("garbage"))) as urlopen:
            self.assertIsNone(planner.suggest_next_experiment(None, []))
        self.assertEqual(urlopen.call_count, 3)

    def test_single_attempt_when_max_retries_is_one(self):
        planner = _planner(force_code=False, max_retries=1)
        with mock.patch.object(lp.request, "urlopen",
                               return_value=_FakeResponse(_completion("garbage"))) as urlopen:
            self.assertIsNone(planner.suggest_next_experiment(None, []))
        self.assertEqual(urlopen.call_count, 1)


class ImportStrippingTests(unittest.TestCase):
    """gpt-4o-mini writes the habitual torch header despite the contract; the
    sandbox rejects the substring 'import ' outright. Strip exactly the
    pre-injected modules and nothing else."""

    def test_strips_the_habitual_torch_header(self):
        raw = ("import torch\n"
               "import torch.nn as nn\n"
               "import torch.nn.functional as F\n"
               "import numpy as np\n"
               "from torch import nn\n"
               "\n" + CODE)
        code = lp._extract_code(raw)
        self.assertNotIn("import", code)
        self.assertTrue(code.startswith("class CandidateModel"))

    def test_keeps_disallowed_imports_so_the_sandbox_still_rejects_them(self):
        code = lp._extract_code("import os\nimport subprocess\n" + CODE)
        self.assertIn("import os", code)
        self.assertIn("import subprocess", code)

    def test_does_not_strip_indented_imports(self):
        code = lp._extract_code("class CandidateModel(nn.Module):\n    import torch\n")
        self.assertIn("import torch", code)


class ValidatorTests(unittest.TestCase):
    def test_rejected_code_is_retried_with_the_compile_error_fed_back(self):
        planner = _planner(force_code=True, max_retries=2)
        bad = json.dumps({"proposal": {"mode": "code", "code": "class CandidateModel(nn.Module):\n    pass"}})
        good = json.dumps({"proposal": {"mode": "code", "code": CODE}})
        seen = {"n": 0}

        def _validator(code):
            seen["n"] += 1
            if seen["n"] == 1:
                raise ValueError("forward() must return a (batch,) tensor; got 2")

        planner.code_validator = _validator
        sent = []

        def _capture(req, **kwargs):
            sent.append(json.loads(req.data.decode("utf-8")))
            return _FakeResponse(_completion(bad if len(sent) == 1 else good))

        with mock.patch.object(lp.request, "urlopen", side_effect=_capture):
            spec = planner.suggest_next_experiment(None, [])
        self.assertIsNotNone(spec)
        self.assertEqual(seen["n"], 2)
        self.assertIn("must return a (batch,) tensor", sent[1]["messages"][1]["content"])

    def test_persistently_bad_code_yields_no_proposal(self):
        planner = _planner(force_code=True, max_retries=2)
        planner.code_validator = mock.Mock(side_effect=ValueError("disallowed token"))
        payload = json.dumps({"proposal": {"mode": "code", "code": CODE}})
        with mock.patch.object(lp.request, "urlopen",
                               return_value=_FakeResponse(_completion(payload))):
            self.assertIsNone(planner.suggest_next_experiment(None, []))
        self.assertIn("generated code rejected", planner.last_error)


if __name__ == "__main__":
    unittest.main()
