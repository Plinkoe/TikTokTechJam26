"""Tests for the multi-stage planner: feature stage, training stage, budgets.

The feature-stage tests matter most: a transform that uses column statistics
would inflate validation and collapse on the hidden test set, and that is
exactly the kind of bug a metric alone will not reveal.
"""

import json
import unittest

import numpy as np

from llm_model_experiment import (CandidateCodeError, compile_feature_transform,
                                  _apply_feature_transform, _make_loss, _make_scheduler,
                                  DEFAULT_CANDIDATE_CODE, compile_candidate_model)
import llm_planner as lp


ROWWISE = """
def transform_features(cats, dense):
    ratio = (dense[:, 0:1] + 1.0) / (np.abs(dense[:, 1:2]) + 1.0)
    return np.concatenate([dense, ratio], axis=1).astype(np.float32)
"""

LEAKY = """
def transform_features(cats, dense):
    centered = dense - dense.mean(axis=0, keepdims=True)
    return centered.astype(np.float32)
"""


class FeatureStageTests(unittest.TestCase):
    def test_rowwise_transform_is_accepted(self):
        fn = compile_feature_transform(ROWWISE)
        out = fn(np.zeros((5, 10), dtype=np.int64), np.ones((5, 10), dtype=np.float32))
        self.assertEqual(np.asarray(out).shape, (5, 11))

    def test_column_statistics_are_rejected_as_leakage(self):
        with self.assertRaises(CandidateCodeError) as ctx:
            compile_feature_transform(LEAKY)
        self.assertIn("not row-wise", str(ctx.exception))

    def test_percentile_normalisation_is_rejected(self):
        with self.assertRaises(CandidateCodeError):
            compile_feature_transform(
                "def transform_features(cats, dense):\n"
                "    return (dense - np.percentile(dense, 50, axis=0)).astype(np.float32)\n")

    def test_cumulative_sum_is_rejected(self):
        with self.assertRaises(CandidateCodeError):
            compile_feature_transform(
                "def transform_features(cats, dense):\n"
                "    return np.cumsum(dense, axis=0).astype(np.float32)\n")

    def test_sorting_across_rows_is_rejected(self):
        with self.assertRaises(CandidateCodeError):
            compile_feature_transform(
                "def transform_features(cats, dense):\n"
                "    return np.sort(dense, axis=0).astype(np.float32)\n")

    def test_imports_are_rejected(self):
        with self.assertRaises(CandidateCodeError) as ctx:
            compile_feature_transform("import os\ndef transform_features(cats, dense):\n    return dense\n")
        self.assertIn("disallowed token", str(ctx.exception))

    def test_wrong_row_count_is_rejected(self):
        with self.assertRaises(CandidateCodeError):
            compile_feature_transform(
                "def transform_features(cats, dense):\n    return dense[:3]\n")

    def test_nan_output_is_rejected(self):
        with self.assertRaises(CandidateCodeError):
            compile_feature_transform(
                "def transform_features(cats, dense):\n"
                "    return (dense / 0.0).astype(np.float32)\n")

    def test_too_many_columns_is_rejected(self):
        with self.assertRaises(CandidateCodeError):
            compile_feature_transform(
                "def transform_features(cats, dense):\n"
                "    return np.tile(dense, (1, 20)).astype(np.float32)\n")

    def test_ndarray_methods_work_inside_the_sandbox(self):
        # numpy lazily imports its method implementations at call time; a
        # sandbox with no __import__ turns an ordinary row-wise feature into
        # KeyError: '__import__'.
        fn = compile_feature_transform(
            "def transform_features(cats, dense):\n"
            "    total = dense.sum(axis=1, keepdims=True)\n"
            "    return np.concatenate([dense, total], axis=1).astype(np.float32)\n")
        out = np.asarray(fn(np.zeros((6, 10), dtype=np.int64), np.ones((6, 10), dtype=np.float32)))
        self.assertEqual(out.shape, (6, 11))
        np.testing.assert_allclose(out[:, -1], 10.0)

    def test_sandbox_still_blocks_dangerous_modules(self):
        from llm_model_experiment import _guarded_import
        for name in ("os", "subprocess", "socket", "importlib"):
            with self.subTest(module=name):
                with self.assertRaises(CandidateCodeError):
                    _guarded_import(name)

    def test_sandbox_blocks_modules_not_already_loaded(self):
        from llm_model_experiment import _guarded_import
        with self.assertRaises(CandidateCodeError):
            _guarded_import("this_module_does_not_exist_anywhere")

    def test_apply_is_a_noop_without_a_transform(self):
        dense = np.ones((4, 10), dtype=np.float32)
        self.assertIs(_apply_feature_transform(None, None, dense), dense)


class TrainingStageTests(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch = torch
        self.logits = torch.tensor([2.0, -1.0, 0.5, -3.0])
        self.y = torch.tensor([1.0, 0.0, 1.0, 0.0])

    def test_each_loss_returns_a_finite_scalar(self):
        for params in ({"loss": "bce"},
                       {"loss": "weighted_bce", "pos_weight": 3.0},
                       {"loss": "focal", "focal_gamma": 2.0, "focal_alpha": 0.25}):
            with self.subTest(**params):
                value = _make_loss(params)(self.logits, self.y)
                self.assertTrue(self.torch.isfinite(value))
                self.assertEqual(value.dim(), 0)

    def test_pos_weight_raises_the_cost_of_missing_positives(self):
        plain = _make_loss({"loss": "bce"})(self.logits, self.y)
        weighted = _make_loss({"loss": "weighted_bce", "pos_weight": 5.0})(self.logits, self.y)
        self.assertGreater(float(weighted), float(plain))

    def test_unknown_loss_is_rejected(self):
        with self.assertRaises(CandidateCodeError):
            _make_loss({"loss": "hinge"})

    def test_schedulers(self):
        model = compile_candidate_model(DEFAULT_CANDIDATE_CODE)([5] * 10, 10)
        opt = self.torch.optim.AdamW(model.parameters(), lr=1e-3)
        self.assertIsNone(_make_scheduler(opt, {"scheduler": "none"}, 5))
        self.assertIsNotNone(_make_scheduler(opt, {"scheduler": "cosine"}, 5))
        self.assertIsNotNone(_make_scheduler(opt, {"scheduler": "step"}, 5))
        with self.assertRaises(CandidateCodeError):
            _make_scheduler(opt, {"scheduler": "magic"}, 5)


class ReferenceArchitectureTests(unittest.TestCase):
    def test_default_candidate_compiles_and_passes_the_dry_run(self):
        # Feature- and training-stage proposals are scored against this, so if
        # it ever stops compiling every non-model stage silently dies.
        self.assertTrue(callable(compile_candidate_model(DEFAULT_CANDIDATE_CODE)))


class ProposalModeTests(unittest.TestCase):
    def test_features_mode_requires_feature_code(self):
        with self.assertRaises(ValueError):
            lp.normalize_proposal({"mode": "features", "params": {}})

    def test_features_mode_normalizes(self):
        p = lp.normalize_proposal({"mode": "features", "feature_code": ROWWISE})
        self.assertEqual(p["family"], "llm_features")
        self.assertIn("transform_features", p["feature_code"])

    def test_train_mode_requires_params(self):
        with self.assertRaises(ValueError):
            lp.normalize_proposal({"mode": "train", "params": {}})

    def test_train_mode_normalizes(self):
        p = lp.normalize_proposal({"mode": "train", "params": {"loss": "focal", "epochs": 4}})
        self.assertEqual(p["family"], "llm_train")
        self.assertEqual(p["params"]["loss"], "focal")

    def test_unknown_mode_is_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            lp.normalize_proposal({"mode": "teleport", "name": "x"})
        self.assertIn("unsupported proposal mode", str(ctx.exception))

    def test_feature_validator_rejection_triggers_a_retry(self):
        from unittest import mock
        cfg = lp.LLMPlannerConfig(enabled=True, api_key="k", base_url="https://x.invalid/v1",
                                  max_retries=2)
        planner = lp.LLMPlanner(cfg)
        planner.feature_validator = compile_feature_transform
        bad = json.dumps({"proposal": {"mode": "features", "feature_code": LEAKY}})

        class _R:
            def __init__(self, body): self._b = json.dumps(body).encode()
            def read(self): return self._b
            def __enter__(self): return self
            def __exit__(self, *a): return False

        body = {"choices": [{"finish_reason": "stop", "message": {"content": bad}}], "usage": {}}
        with mock.patch.object(lp.request, "urlopen", return_value=_R(body)) as urlopen:
            self.assertIsNone(planner.suggest_next_experiment(None, []))
        self.assertEqual(urlopen.call_count, 2)
        self.assertIn("feature transform rejected", planner.last_error)


if __name__ == "__main__":
    unittest.main()
