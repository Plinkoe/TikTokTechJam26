"""Entry point for the autonomous KuaiRand research agent.

Set KUAI_LLM_CODE_MODE=true to let the LLM CREATE executable experiment modules
instead of selecting only from the pre-written registry. The generated mode is
validation-only during research and keeps the original controller as the fallback.
"""
import os

from agent_architecture import BenchmarkController, build_parser
from autonomous_code_agent import AutonomousCodeAgent


def main() -> None:
    args = build_parser().parse_args()
    controller = BenchmarkController(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        max_iters=args.max_iters,
        epsilon=args.epsilon,
        patience=args.patience,
        seed=args.seed,
    )

    code_mode = os.getenv("KUAI_LLM_CODE_MODE", "false").strip().lower() in {"1", "true", "yes", "on"}
    if code_mode and controller.llm_planner.enabled:
        result = AutonomousCodeAgent(controller, max_iters=args.max_iters).run()
        print("\n=== autonomous LLM code-research agent ===")
        print(f"baseline validation primary: {result['baseline_primary']:.6f}")
        print(f"best generated validation primary: {result['best_validation_primary']:.6f}")
        print(f"improved over baseline: {result['improved_over_baseline']}")
        print(f"research log: {os.path.join(args.out_dir, 'iterations.jsonl')}")
        return

    result = controller.run()
    submission_path = controller.finalize_submission(args.submission_path)
    print("\n=== autonomous research agent ===")
    print(f"best validation model: {result['best_model']}")
    print(f"best validation metrics: {result['best']}")
    print(f"submission: {submission_path}")


if __name__ == "__main__":
    main()
