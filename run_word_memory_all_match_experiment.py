"""Run one-word memory experiment with all-previous-word match feedback."""

from __future__ import annotations

import argparse

from run_word_memory_feedback_experiment import run_experiment

CONVERSATION_SEED = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=["openrouter", "anthropic"], default="openrouter")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_experiment(
        experiment_name="word_memory_feedback",
        model="anthropic/claude-opus-4.6",
        steps=30,
        feedback_mode="all_previous",
        new_run=True,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        reasoning_effort="none",
        conversation_seed=CONVERSATION_SEED,
        provider=args.provider,
    )


if __name__ == "__main__":
    main()
