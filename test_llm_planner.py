import os
import tempfile
import unittest

from llm_planner import LLMPlannerConfig, load_dotenv, normalize_proposal, resolve_experiment_spec


class LLMPlannerTests(unittest.TestCase):
    def test_disabled_config_is_off(self):
        cfg = LLMPlannerConfig.disabled()
        self.assertFalse(cfg.enabled)

    def test_normalize_proposal_accepts_valid_json(self):
        proposal = normalize_proposal({
            "name": "fm_baseline",
            "params": {"k": 16, "lr": 0.001},
            "hypothesis": "Reproduce the baseline.",
            "code_diff": "baseline hyperparams"
        })
        self.assertEqual(proposal["name"], "fm_baseline")
        self.assertEqual(proposal["params"]["k"], 16)

    def test_resolve_experiment_uses_known_registry_name(self):
        spec = resolve_experiment_spec({
            "name": "history_deepfm",
            "params": {"epochs": 4},
            "hypothesis": "Try a lighter history model.",
            "code_diff": "reduce epochs"
        })
        self.assertEqual(spec.name, "history_deepfm")
        self.assertEqual(spec.params["epochs"], 4)

    def test_load_dotenv_reads_values(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            env_path = os.path.join(tmpdir, ".env")
            with open(env_path, "w", encoding="utf-8") as fh:
                fh.write("KUAI_LLM_ENABLED=true\n")
                fh.write("KUAI_LLM_MODEL=gpt-4o-mini\n")
                fh.write("KUAI_LLM_API_KEY=test-key\n")
            os.environ.pop("KUAI_LLM_ENABLED", None)
            os.environ.pop("KUAI_LLM_MODEL", None)
            os.environ.pop("KUAI_LLM_API_KEY", None)
            self.assertTrue(load_dotenv(env_path))
            self.assertEqual(os.environ["KUAI_LLM_ENABLED"], "true")
            self.assertEqual(os.environ["KUAI_LLM_MODEL"], "gpt-4o-mini")
            self.assertEqual(os.environ["KUAI_LLM_API_KEY"], "test-key")


if __name__ == "__main__":
    unittest.main()
