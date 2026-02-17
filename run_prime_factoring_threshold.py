"""Find prime-factoring failure threshold with logarithmic binary search."""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
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


DEFAULT_EXPERIMENT_NAME = "prime_factor_threshold"


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
        is_witness = True
        for _ in range(s - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                is_witness = False
                break
        if is_witness:
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


def generate_semiprime_with_digits(total_digits: int, rng: random.Random) -> tuple[int, list[int]]:
    if total_digits < 2:
        raise ValueError("total_digits must be >= 2")

    p_digits = total_digits // 2
    q_digits = total_digits - p_digits
    if p_digits < 1:
        p_digits = 1
    if q_digits < 1:
        q_digits = 1

    while True:
        p = generate_prime_with_digits(p_digits, rng)
        q = generate_prime_with_digits(q_digits, rng)
        n = p * q
        if len(str(n)) == total_digits:
            return n, sorted([p, q])


def _pollard_rho_factor(n: int, rng: random.Random, start_time: float, timeout_seconds: float) -> int:
    if n % 2 == 0:
        return 2
    if is_probable_prime(n):
        return n

    while True:
        if time.perf_counter() - start_time > timeout_seconds:
            raise TimeoutError("Classical factoring timed out.")
        x = rng.randrange(2, n - 1)
        y = x
        c = rng.randrange(1, n - 1)
        d = 1
        while d == 1:
            if time.perf_counter() - start_time > timeout_seconds:
                raise TimeoutError("Classical factoring timed out.")
            x = (pow(x, 2, n) + c) % n
            y = (pow(y, 2, n) + c) % n
            y = (pow(y, 2, n) + c) % n
            d = math.gcd(abs(x - y), n)
        if d != n:
            return d


def factor_integer_with_timeout(n: int, rng: random.Random, timeout_seconds: float) -> list[int] | None:
    start_time = time.perf_counter()
    factors: list[int] = []
    stack = [n]
    try:
        while stack:
            if time.perf_counter() - start_time > timeout_seconds:
                return None
            value = stack.pop()
            if value == 1:
                continue
            if is_probable_prime(value):
                factors.append(value)
                continue
            divisor = _pollard_rho_factor(value, rng, start_time, timeout_seconds)
            if divisor == value:
                factors.append(value)
            else:
                stack.append(divisor)
                stack.append(value // divisor)
    except TimeoutError:
        return None
    return sorted(factors)


def can_classically_factor_digits(
    digits: int,
    timeout_seconds: float,
    samples: int,
    base_seed: int,
) -> bool:
    for sample_idx in range(samples):
        rng = random.Random(base_seed + (digits * 10_000) + sample_idx)
        n, true_factors = generate_semiprime_with_digits(digits, rng)
        factor_rng = random.Random(base_seed + 99_999 + (digits * 10_000) + sample_idx)
        guessed = factor_integer_with_timeout(n, factor_rng, timeout_seconds)
        if guessed is None or guessed != sorted(true_factors):
            return False
    return True


def find_classical_upper_bound_digits(
    min_digits: int,
    max_digits: int,
    timeout_seconds: float,
    samples: int,
    base_seed: int,
) -> int | None:
    if not can_classically_factor_digits(min_digits, timeout_seconds, samples, base_seed):
        return None
    if can_classically_factor_digits(max_digits, timeout_seconds, samples, base_seed):
        return max_digits

    lo = min_digits
    hi = max_digits
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if can_classically_factor_digits(mid, timeout_seconds, samples, base_seed):
            lo = mid
        else:
            hi = mid - 1
    return lo


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
        raw_factors = parsed["factors"]
        factors: list[int] = []
        for item in raw_factors:
            if isinstance(item, int):
                factors.append(item)
            elif isinstance(item, str) and item.strip().isdigit():
                factors.append(int(item.strip()))
            else:
                return None
        if factors:
            return factors

    bracket_match = re.search(r"\[\s*([0-9,\s]+)\s*\]", content)
    if not bracket_match:
        return None
    raw_items = [token.strip() for token in bracket_match.group(1).split(",") if token.strip()]
    try:
        factors = [int(token) for token in raw_items]
    except ValueError:
        return None
    return factors if factors else None


def factor_guess_is_correct(guess: list[int] | None, true_factors: list[int], n: int) -> bool:
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


def init_or_resume_run(
    experiment_name: str,
    model: str,
    min_digits: int,
    max_digits: int,
    trials_per_size: int,
    pass_rate: float,
    seed: int,
    enforce_classical_upper_bound: bool,
    classical_timeout_seconds: float,
    classical_bound_samples: int,
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
                or int(state["min_digits"]) != int(min_digits)
                or int(state["max_digits"]) != int(max_digits)
                or int(state["trials_per_size"]) != int(trials_per_size)
                or float(state["pass_rate"]) != float(pass_rate)
                or int(state["seed"]) != int(seed)
                or bool(state.get("enforce_classical_upper_bound", True)) != bool(enforce_classical_upper_bound)
                or float(state.get("classical_timeout_seconds", 1.0)) != float(classical_timeout_seconds)
                or int(state.get("classical_bound_samples", 1)) != int(classical_bound_samples)
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
        "min_digits": min_digits,
        "max_digits": max_digits,
        "trials_per_size": trials_per_size,
        "pass_rate": pass_rate,
        "seed": seed,
        "enforce_classical_upper_bound": enforce_classical_upper_bound,
        "classical_timeout_seconds": classical_timeout_seconds,
        "classical_bound_samples": classical_bound_samples,
        "started_at": now_iso(),
        "updated_at": now_iso(),
        "status": "running",
        "evaluations": {},
    }
    write_state(run_dir / STATE_FILE_NAME, state)
    append_event(run_dir, "run_created", {"run_id": run_id})
    return run_dir, state


def load_trial_results(steps_dir: Path) -> dict[int, list[dict[str, Any]]]:
    by_digits: dict[int, list[dict[str, Any]]] = {}
    for step_file in sorted(steps_dir.glob("*.json")):
        try:
            step = json.loads(step_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if step.get("status") != "completed":
            continue
        digits = step.get("digits")
        if not isinstance(digits, int):
            continue
        by_digits.setdefault(digits, []).append(step)
    return by_digits


def summarize_digit_result(results: list[dict[str, Any]], pass_rate: float) -> dict[str, Any]:
    total = len(results)
    correct = sum(1 for item in results if item.get("is_correct") is True)
    accuracy = (correct / total) if total else 0.0
    passed = accuracy >= pass_rate
    return {
        "total_trials": total,
        "correct_trials": correct,
        "accuracy": accuracy,
        "passed": passed,
    }


def evaluate_digits(
    digits: int,
    run_dir: Path,
    steps_dir: Path,
    trials_per_size: int,
    pass_rate: float,
    client: OpenRouterClient,
    rng: random.Random,
    existing_by_digits: dict[int, list[dict[str, Any]]],
) -> dict[str, Any]:
    existing = existing_by_digits.get(digits, [])
    existing_by_trial = {
        int(item["trial_index"]): item
        for item in existing
        if isinstance(item.get("trial_index"), int)
    }
    results: list[dict[str, Any]] = list(existing)

    for trial_index in range(1, trials_per_size + 1):
        if trial_index in existing_by_trial:
            continue

        n, true_factors = generate_semiprime_with_digits(digits, rng)
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
        is_correct = factor_guess_is_correct(guess, true_factors, n)

        record = {
            "status": "completed",
            "completed_at": now_iso(),
            "digits": digits,
            "trial_index": trial_index,
            "n": str(n),
            "true_factors": [str(item) for item in true_factors],
            "model_guess": [str(item) for item in guess] if guess is not None else None,
            "is_correct": is_correct,
            "prompt": {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
            },
            "raw_response": response_json,
        }
        step_file = steps_dir / f"d{digits:04d}_t{trial_index:04d}.json"
        step_file.write_text(json.dumps(record, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
        append_event(
            run_dir,
            "trial_completed",
            {
                "digits": digits,
                "trial_index": trial_index,
                "is_correct": is_correct,
                "step_file": str(step_file),
            },
        )
        print(f"digits={digits} trial={trial_index}/{trials_per_size} correct={is_correct}")
        results.append(record)

    existing_by_digits[digits] = results
    return summarize_digit_result(results, pass_rate)


def run_threshold_search(
    experiment_name: str,
    model: str,
    min_digits: int,
    max_digits: int,
    trials_per_size: int,
    pass_rate: float,
    seed: int,
    enforce_classical_upper_bound: bool,
    classical_timeout_seconds: float,
    classical_bound_samples: int,
    new_run: bool,
    temperature: float,
    top_p: float,
    max_tokens: int | None,
) -> None:
    run_dir, state = init_or_resume_run(
        experiment_name=experiment_name,
        model=model,
        min_digits=min_digits,
        max_digits=max_digits,
        trials_per_size=trials_per_size,
        pass_rate=pass_rate,
        seed=seed,
        enforce_classical_upper_bound=enforce_classical_upper_bound,
        classical_timeout_seconds=classical_timeout_seconds,
        classical_bound_samples=classical_bound_samples,
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
    existing_by_digits = load_trial_results(steps_dir)

    effective_max_digits = max_digits
    if enforce_classical_upper_bound:
        append_event(
            run_dir,
            "classical_bound_search_started",
            {
                "min_digits": min_digits,
                "max_digits": max_digits,
                "timeout_seconds": classical_timeout_seconds,
                "samples": classical_bound_samples,
            },
        )
        classical_upper_bound = find_classical_upper_bound_digits(
            min_digits=min_digits,
            max_digits=max_digits,
            timeout_seconds=classical_timeout_seconds,
            samples=classical_bound_samples,
            base_seed=seed,
        )
        if classical_upper_bound is None:
            raise RuntimeError(
                "Classical factoring could not solve min_digits within timeout. "
                "Lower --min-digits or increase --classical-timeout-seconds."
            )
        effective_max_digits = min(max_digits, classical_upper_bound)
        append_event(
            run_dir,
            "classical_bound_search_completed",
            {
                "classical_upper_bound_digits": classical_upper_bound,
                "effective_max_digits": effective_max_digits,
            },
        )
        print(
            f"classical_upper_bound_digits={classical_upper_bound} "
            f"(effective max_digits={effective_max_digits})"
        )
        if effective_max_digits < min_digits:
            raise RuntimeError(
                "Effective max digits fell below min digits after classical bound check. "
                "Lower --min-digits or disable bound with --no-classical-upper-bound."
            )

    memo: dict[int, dict[str, Any]] = {}

    def get_result(digits: int) -> dict[str, Any]:
        if digits in memo:
            return memo[digits]
        append_event(run_dir, "digits_evaluation_started", {"digits": digits})
        summary = evaluate_digits(
            digits=digits,
            run_dir=run_dir,
            steps_dir=steps_dir,
            trials_per_size=trials_per_size,
            pass_rate=pass_rate,
            client=client,
            rng=rng,
            existing_by_digits=existing_by_digits,
        )
        append_event(run_dir, "digits_evaluation_completed", {"digits": digits, **summary})
        memo[digits] = summary
        state_evals = state.get("evaluations", {})
        state_evals[str(digits)] = summary
        state["evaluations"] = state_evals
        write_state(run_dir / STATE_FILE_NAME, state)
        print(
            f"digits={digits} accuracy={summary['accuracy']:.3f} "
            f"({summary['correct_trials']}/{summary['total_trials']}) passed={summary['passed']}"
        )
        return summary

    append_event(run_dir, "search_started", {})

    lo = min_digits
    hi = effective_max_digits
    first_fail: int | None = None
    while lo <= hi:
        mid = (lo + hi) // 2
        mid_result = get_result(mid)
        if mid_result["passed"]:
            lo = mid + 1
        else:
            first_fail = mid
            hi = mid - 1

    if first_fail is None:
        result = {
            "first_failing_digits": None,
            "largest_passing_digits": effective_max_digits,
            "failing_digits_accuracy": None,
            "passing_digits_accuracy": get_result(effective_max_digits)["accuracy"],
            "trials_per_size": trials_per_size,
            "pass_rate": pass_rate,
            "message": f"No failure found up to max_digits={effective_max_digits}.",
        }
        state["status"] = "completed"
        state["result"] = result
        write_state(run_dir / STATE_FILE_NAME, state)
        append_event(run_dir, "search_completed", result)
        print(f"No failure found up to {effective_max_digits} digits.")
        print(f"Run complete: {run_dir}")
        return

    prior_pass = first_fail - 1
    prior_pass_result = get_result(prior_pass) if prior_pass >= min_digits else None
    fail_result = get_result(first_fail)
    result = {
        "first_failing_digits": first_fail,
        "largest_passing_digits": prior_pass if prior_pass >= min_digits else None,
        "failing_digits_accuracy": fail_result["accuracy"],
        "passing_digits_accuracy": prior_pass_result["accuracy"] if prior_pass_result else None,
        "trials_per_size": trials_per_size,
        "pass_rate": pass_rate,
    }
    state["status"] = "completed"
    state["result"] = result
    write_state(run_dir / STATE_FILE_NAME, state)
    append_event(run_dir, "search_completed", result)

    print(f"First failing size: {first_fail} digits")
    if prior_pass_result is not None:
        print(f"Largest passing size: {prior_pass} digits")
    print(f"Run complete: {run_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Use logarithmic binary search to estimate when LLM prime factoring starts failing."
    )
    parser.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT_NAME, help="Experiment output namespace.")
    parser.add_argument("--new-run", action="store_true", help="Start new run instead of resuming latest.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="OpenRouter model ID.")
    parser.add_argument("--min-digits", type=int, default=4, help="Smallest digit length to test.")
    parser.add_argument("--max-digits", type=int, default=40, help="Largest digit length to test.")
    parser.add_argument("--trials-per-size", type=int, default=2, help="Trials per digit length.")
    parser.add_argument("--pass-rate", type=float, default=1.0, help="Minimum accuracy needed to mark a size as pass.")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed for generated semiprimes.")
    parser.add_argument(
        "--no-classical-upper-bound",
        action="store_true",
        help="Disable pre-check that clamps max digits by a 1s classical factoring bound.",
    )
    parser.add_argument(
        "--classical-timeout-seconds",
        type=float,
        default=1.0,
        help="Timeout used for classical factoring upper-bound check.",
    )
    parser.add_argument(
        "--classical-bound-samples",
        type=int,
        default=1,
        help="How many random semiprimes per digit size for classical bound check.",
    )
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    parser.add_argument("--max-tokens", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.min_digits < 2:
        raise ValueError("--min-digits must be >= 2")
    if args.max_digits < args.min_digits:
        raise ValueError("--max-digits must be >= --min-digits")
    if args.trials_per_size < 1:
        raise ValueError("--trials-per-size must be >= 1")
    if not (0.0 <= args.pass_rate <= 1.0):
        raise ValueError("--pass-rate must be in [0.0, 1.0]")
    if args.classical_timeout_seconds <= 0:
        raise ValueError("--classical-timeout-seconds must be > 0")
    if args.classical_bound_samples < 1:
        raise ValueError("--classical-bound-samples must be >= 1")

    run_threshold_search(
        experiment_name=args.experiment_name,
        model=args.model,
        min_digits=args.min_digits,
        max_digits=args.max_digits,
        trials_per_size=args.trials_per_size,
        pass_rate=args.pass_rate,
        seed=args.seed,
        enforce_classical_upper_bound=not args.no_classical_upper_bound,
        classical_timeout_seconds=args.classical_timeout_seconds,
        classical_bound_samples=args.classical_bound_samples,
        new_run=args.new_run,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )


if __name__ == "__main__":
    main()
