"""Evaluate factorization accuracy across integer partitions of a target sum."""

from __future__ import annotations

import argparse
import json
import random
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from eval_constants import (
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    EVENTS_FILE_NAME,
    EXPERIMENT_OUTPUT_DIR,
    RUNS_DIR_NAME,
    STATE_FILE_NAME,
    STEPS_DIR_NAME,
)
from openrouter_client import OpenRouterClient, OpenRouterConfig


DEFAULT_EXPERIMENT_NAME = "partition_factorization"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_event(run_dir: Path, event_type: str, payload: dict[str, Any]) -> None:
    line = {"ts": now_iso(), "event": event_type, **payload}
    events_path = run_dir / EVENTS_FILE_NAME
    with events_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(line, ensure_ascii=True) + "\n")


def read_state(state_path: Path) -> dict[str, Any]:
    return json.loads(state_path.read_text(encoding="utf-8"))


def write_state(state_path: Path, state: dict[str, Any]) -> None:
    state["updated_at"] = now_iso()
    state_path.write_text(json.dumps(state, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def latest_run_dir(runs_dir: Path) -> Path | None:
    candidates = [p for p in runs_dir.iterdir() if p.is_dir()]
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: p.name)[-1]


def is_probable_prime(n: int, rounds: int = 12) -> bool:
    if n < 2:
        return False
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
    for p in small_primes:
        if n == p:
            return True
        if n % p == 0:
            return False

    d = n - 1
    s = 0
    while d % 2 == 0:
        s += 1
        d //= 2

    for _ in range(rounds):
        a = random.randrange(2, n - 1)
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        witness = True
        for _ in range(s - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                witness = False
                break
        if witness:
            return False
    return True


def generate_prime_with_digits(digits: int, rng: random.Random) -> int:
    if digits < 1:
        raise ValueError("digits must be >= 1")
    if digits == 1:
        return rng.choice([2, 3, 5, 7])

    lower = 10 ** (digits - 1)
    upper = (10**digits) - 1
    while True:
        candidate = rng.randrange(lower, upper + 1)
        if candidate % 2 == 0:
            candidate += 1
        if candidate > upper:
            candidate -= 2
        if candidate < lower:
            continue
        if is_probable_prime(candidate):
            return candidate


def partitions_of(n: int, max_part: int | None = None) -> list[list[int]]:
    if n == 0:
        return [[]]
    if n < 0:
        return []
    if max_part is None:
        max_part = n

    out: list[list[int]] = []
    for first in range(min(max_part, n), 1 - 1, -1):
        for rest in partitions_of(n - first, first):
            out.append([first] + rest)
    return out


def partition_key(parts: list[int]) -> str:
    return "-".join(str(p) for p in parts)


def extract_json_object(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    for match in re.finditer(r"(?s)\{.*?\}", text):
        chunk = match.group(0)
        try:
            parsed = json.loads(chunk)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue
    return None


def parse_factor_guess(content: str) -> list[int] | None:
    parsed = extract_json_object(content)
    if parsed is not None and isinstance(parsed.get("factors"), list):
        factors: list[int] = []
        for item in parsed["factors"]:
            if isinstance(item, int):
                factors.append(item)
            elif isinstance(item, str) and item.strip().isdigit():
                factors.append(int(item.strip()))
            else:
                return None
        return factors if factors else None

    bracket_match = re.search(r"\[\s*([0-9,\s]+)\s*\]", content)
    if not bracket_match:
        return None
    raw = [token.strip() for token in bracket_match.group(1).split(",") if token.strip()]
    try:
        factors = [int(token) for token in raw]
    except ValueError:
        return None
    return factors if factors else None


def factors_correct(guess: list[int] | None, true_factors: list[int], n: int) -> bool:
    if guess is None:
        return False
    if sorted(guess) != sorted(true_factors):
        return False
    if any(item <= 1 or not is_probable_prime(item) for item in guess):
        return False
    product = 1
    for item in guess:
        product *= item
    return product == n


def load_trial_results(steps_dir: Path) -> dict[str, list[dict[str, Any]]]:
    by_partition: dict[str, list[dict[str, Any]]] = {}
    for step_file in sorted(steps_dir.glob("*.json")):
        try:
            step = json.loads(step_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if step.get("status") != "completed":
            continue
        key = step.get("partition_key")
        if not isinstance(key, str):
            continue
        by_partition.setdefault(key, []).append(step)
    return by_partition


def summarize_partition_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    correct = sum(1 for r in results if r.get("is_correct") is True)
    accuracy = (correct / total) if total else 0.0
    return {
        "trials": total,
        "correct": correct,
        "accuracy": accuracy,
    }


def init_or_resume_run(
    experiment_name: str,
    model: str,
    partition_sum: int,
    trials_per_partition: int,
    seed: int,
    force_new_run: bool,
) -> tuple[Path, dict[str, Any]]:
    experiment_root = Path(EXPERIMENT_OUTPUT_DIR) / experiment_name
    runs_dir = experiment_root / RUNS_DIR_NAME
    runs_dir.mkdir(parents=True, exist_ok=True)

    if not force_new_run:
        latest = latest_run_dir(runs_dir)
        if latest is not None:
            state_path = latest / STATE_FILE_NAME
            if not state_path.exists():
                raise FileNotFoundError(f"Missing state file in latest run: {state_path}")
            state = read_state(state_path)
            if (
                state["model"] != model
                or int(state["partition_sum"]) != int(partition_sum)
                or int(state["trials_per_partition"]) != int(trials_per_partition)
                or int(state["seed"]) != int(seed)
            ):
                raise ValueError("Latest run uses different config. Use --new-run or pass the same config.")
            return latest, state

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = runs_dir / run_id
    (run_dir / STEPS_DIR_NAME).mkdir(parents=True, exist_ok=True)
    state = {
        "run_id": run_id,
        "experiment": experiment_name,
        "model": model,
        "partition_sum": partition_sum,
        "trials_per_partition": trials_per_partition,
        "seed": seed,
        "started_at": now_iso(),
        "updated_at": now_iso(),
        "status": "running",
        "summary": {},
    }
    write_state(run_dir / STATE_FILE_NAME, state)
    append_event(run_dir, "run_created", {"run_id": run_id})
    return run_dir, state


def run_partition_experiment(
    experiment_name: str,
    model: str,
    partition_sum: int,
    trials_per_partition: int,
    seed: int,
    new_run: bool,
    temperature: float,
    top_p: float,
    max_tokens: int | None,
) -> None:
    partitions = partitions_of(partition_sum)
    partitions = [parts for parts in partitions if 1 not in parts]
    partitions = sorted(partitions, reverse=True)
    partition_keys = [partition_key(p) for p in partitions]

    run_dir, state = init_or_resume_run(
        experiment_name=experiment_name,
        model=model,
        partition_sum=partition_sum,
        trials_per_partition=trials_per_partition,
        seed=seed,
        force_new_run=new_run,
    )
    steps_dir = run_dir / STEPS_DIR_NAME
    steps_dir.mkdir(parents=True, exist_ok=True)

    client = OpenRouterClient(
        OpenRouterConfig(
            model=model,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
    )
    rng = random.Random(seed)
    existing = load_trial_results(steps_dir)
    summary: dict[str, Any] = state.get("summary", {})

    append_event(run_dir, "experiment_started", {"partition_count": len(partitions)})

    for idx, parts in enumerate(partitions, start=1):
        key = partition_key(parts)
        existing_trials = existing.get(key, [])
        by_trial = {
            int(row["trial_index"]): row
            for row in existing_trials
            if isinstance(row.get("trial_index"), int)
        }
        results = list(existing_trials)

        append_event(
            run_dir,
            "partition_started",
            {"partition_key": key, "partition": parts, "position": idx, "total_partitions": len(partitions)},
        )
        print(f"partition {idx}/{len(partitions)}: {key}")

        for trial_index in range(1, trials_per_partition + 1):
            if trial_index in by_trial:
                continue

            true_factors = [generate_prime_with_digits(d, rng) for d in parts]
            n = 1
            for f in true_factors:
                n *= f

            system_prompt = (
                "You are a precise number theory assistant.\n"
                "Prime-factor the given integer.\n"
                "Return ONLY valid JSON of the form {\"factors\":[p1,p2,...]} with integer prime factors."
            )
            user_prompt = f"Factor this integer into primes: {n}"

            response_json = client.chat_completion(user_prompt=user_prompt, system_prompt=system_prompt)
            message = response_json.get("choices", [{}])[0].get("message", {})
            content = message.get("content", "")
            guess = parse_factor_guess(content)
            is_correct = factors_correct(guess, true_factors, n)

            record = {
                "status": "completed",
                "completed_at": now_iso(),
                "partition_sum": partition_sum,
                "partition": parts,
                "partition_key": key,
                "trial_index": trial_index,
                "n": str(n),
                "true_factors": [str(v) for v in true_factors],
                "model_guess": [str(v) for v in guess] if guess is not None else None,
                "is_correct": is_correct,
                "prompt": {
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                },
                "raw_response": response_json,
            }
            step_file = steps_dir / f"p{key}_t{trial_index:04d}.json"
            step_file.write_text(json.dumps(record, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
            append_event(
                run_dir,
                "trial_completed",
                {
                    "partition_key": key,
                    "trial_index": trial_index,
                    "is_correct": is_correct,
                    "step_file": str(step_file),
                },
            )
            print(f"  trial {trial_index}/{trials_per_partition}: correct={is_correct}")
            results.append(record)

        existing[key] = results
        part_summary = summarize_partition_results(results)
        summary[key] = {
            "partition": parts,
            **part_summary,
        }
        state["summary"] = summary
        write_state(run_dir / STATE_FILE_NAME, state)
        append_event(run_dir, "partition_completed", {"partition_key": key, **part_summary})
        print(
            f"  summary {key}: accuracy={part_summary['accuracy']:.3f} "
            f"({part_summary['correct']}/{part_summary['trials']})"
        )

    leaderboard = sorted(
        [
            {
                "partition_key": key,
                "partition": summary[key]["partition"],
                "accuracy": summary[key]["accuracy"],
                "correct": summary[key]["correct"],
                "trials": summary[key]["trials"],
            }
            for key in partition_keys
            if key in summary
        ],
        key=lambda row: (row["accuracy"], row["correct"]),
        reverse=True,
    )
    state["status"] = "completed"
    state["leaderboard"] = leaderboard
    write_state(run_dir / STATE_FILE_NAME, state)
    append_event(run_dir, "experiment_completed", {"partition_count": len(leaderboard)})

    print("\nLeaderboard (best to worst):")
    for row in leaderboard:
        print(f"{row['partition_key']}: accuracy={row['accuracy']:.3f} ({row['correct']}/{row['trials']})")
    print(f"\nRun complete: {run_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate prime factorization accuracy across integer partitions of a target sum."
    )
    parser.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT_NAME, help="Experiment output namespace.")
    parser.add_argument("--new-run", action="store_true", help="Start new run instead of resuming latest.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="OpenRouter model ID.")
    parser.add_argument("--partition-sum", type=int, default=8, help="Integer to partition.")
    parser.add_argument("--trials-per-partition", type=int, default=5, help="Trials for each partition.")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed.")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    parser.add_argument("--max-tokens", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.partition_sum < 1:
        raise ValueError("--partition-sum must be >= 1")
    if args.trials_per_partition < 1:
        raise ValueError("--trials-per-partition must be >= 1")

    run_partition_experiment(
        experiment_name=args.experiment_name,
        model=args.model,
        partition_sum=args.partition_sum,
        trials_per_partition=args.trials_per_partition,
        seed=args.seed,
        new_run=args.new_run,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )


if __name__ == "__main__":
    main()
