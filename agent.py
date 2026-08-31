"""Legacy entry point for the autonomous KuaiRand agent.

This wrapper delegates to the modular controller in agent_architecture.py so the
repo keeps a single, tested implementation for baseline verification, experiment
looping, and final submission generation.
"""

from agent_architecture import BenchmarkController, build_parser


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
    result = controller.run()
    submission_path = controller.finalize_submission(args.submission_path)
    print("\n=== autonomous research agent ===")
    print(f"best validation model: {result['best_model']}")
    print(f"best validation metrics: {result['best']}")
    print(f"submission: {submission_path}")


if __name__ == "__main__":
    main()
