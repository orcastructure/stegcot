"""Run a filler-token sensitivity experiment for arithmetic prompts."""

from __future__ import annotations

import argparse
import json
import random
import re
import string
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openrouter_client import OpenRouterClient, OpenRouterConfig


OUTPUT_ROOT = Path("experiment_outputs") / "filler_token_arithmetic"

BASE_SYSTEM_PROMPT = (
    "You are a precise arithmetic solver. "
    "Return only the final numeric answer with no words or explanation."
)

FILLER_ALPHABET = string.ascii_letters + string.digits + string.punctuation


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_id_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def extract_first_number(text: str) -> str | None:
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return None
    return match.group(0)


def normalize_number(value: str | int | float) -> str:
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f"{value:.12g}"
    raw = str(value).strip()
    try:
        as_float = float(raw)
    except ValueError:
        return raw
    if as_float.is_integer():
        return str(int(as_float))
    return f"{as_float:.12g}"


def is_correct(model_text: str, expected_answer: str | int | float) -> tuple[bool, str | None]:
    first_number = extract_first_number(model_text)
    if first_number is None:
        return False, None
    return normalize_number(first_number) == normalize_number(expected_answer), first_number


def random_filler(length: int, rng: random.Random) -> str:
    return "".join(rng.choice(FILLER_ALPHABET) for _ in range(length))


def compose_prompt(problem_text: str, filler: str | None) -> str:
    base = [f"Solve this arithmetic problem:\n{problem_text.strip()}"]
    if filler is not None:
        base.append(f"Here are some filler tokens: {filler}")
    base.append("Output only the final numeric answer.")
    return "\n\n".join(base)


def safe_eval_int_expression(expr: str) -> int:
    # Controlled local eval for generated integer arithmetic only.
    return int(eval(expr, {"__builtins__": {}}, {}))


def generate_problem(rng: random.Random, steps: int, width: int) -> tuple[str, int]:
    nums = [rng.randint(10 ** (width - 1), (10**width) - 1) for _ in range(steps + 1)]
    ops = [rng.choice(["+", "-", "*"]) for _ in range(steps)]
    expr = str(nums[0])
    for idx, op in enumerate(ops, start=1):
        rhs = nums[idx]
        # Add nested structure so order-of-operations matters.
        if idx % 2 == 0:
            expr = f"({expr} {op} {rhs})"
        else:
            expr = f"{expr} {op} {rhs}"
    answer = safe_eval_int_expression(expr)
    return expr, answer


def load_problems(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict) and isinstance(payload.get("problems"), list):
        rows = payload["problems"]
    else:
        raise ValueError("Problems file must be a list or {'problems': [...]}.")

    out: list[dict[str, Any]] = []
    for idx, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"Problem entry at index {idx} must be an object.")
        text = row.get("text")
        answer = row.get("answer")
        if not isinstance(text, str) or answer is None:
            raise ValueError(f"Problem entry at index {idx} must include text and answer.")
        out.append({"id": row.get("id", idx), "text": text, "answer": answer})
    return out


def generate_problems(count: int, steps: int, width: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    problems: list[dict[str, Any]] = []
    for idx in range(1, count + 1):
        expr, answer = generate_problem(rng=rng, steps=steps, width=width)
        problems.append({"id": idx, "text": expr, "answer": answer, "steps": steps, "width": width})
    return problems


def build_filler_set(
    variants: int,
    filler_length: int,
    seed: int,
    include_control: bool,
    explicit_fillers: list[str] | None,
) -> list[str | None]:
    fillers: list[str | None] = [None] if include_control else []
    if explicit_fillers:
        fillers.extend(explicit_fillers)
        return fillers
    rng = random.Random(seed)
    for _ in range(variants):
        fillers.append(random_filler(filler_length, rng))
    return fillers


@dataclass
class RequestMeta:
    problem_id: Any
    problem_text: str
    expected_answer: Any
    filler_index: int
    filler: str | None
    request_index: int


def ensure_output_run_dir(run_name: str | None) -> Path:
    run_id = run_name or run_id_now()
    run_dir = OUTPUT_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def run_main(args: argparse.Namespace) -> None:
    if args.problems_file:
        problems = load_problems(Path(args.problems_file))
    else:
        problems = generate_problems(
            count=args.problem_count,
            steps=args.steps,
            width=args.width,
            seed=args.problem_seed,
        )

    fillers = build_filler_set(
        variants=args.filler_variants,
        filler_length=args.filler_length,
        seed=args.filler_seed,
        include_control=not args.no_control,
        explicit_fillers=args.filler,
    )

    run_dir = ensure_output_run_dir(args.run_name)
    run_manifest = {
        "created_at": now_iso(),
        "mode": "run",
        "model": args.model,
        "temperature": args.temperature,
        "workers": args.workers,
        "problem_count": len(problems),
        "filler_count": len(fillers),
    }
    (run_dir / "manifest.json").write_text(json.dumps(run_manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    client = OpenRouterClient(
        OpenRouterConfig(
            model=args.model,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
        )
    )

    batch_requests: list[dict[str, Any]] = []
    metadata: list[RequestMeta] = []
    req_idx = 0
    for problem in problems:
        for filler_idx, filler in enumerate(fillers):
            req_idx += 1
            prompt = compose_prompt(problem_text=problem["text"], filler=filler)
            batch_requests.append({"user_prompt": prompt, "system_prompt": BASE_SYSTEM_PROMPT})
            metadata.append(
                RequestMeta(
                    problem_id=problem["id"],
                    problem_text=problem["text"],
                    expected_answer=problem["answer"],
                    filler_index=filler_idx,
                    filler=filler,
                    request_index=req_idx,
                )
            )

    results = client.chat_completion_batch(requests_batch=batch_requests, max_workers=args.workers)

    records: list[dict[str, Any]] = []
    for meta, raw in zip(metadata, results):
        if isinstance(raw, Exception):
            record = {
                "request_index": meta.request_index,
                "problem_id": meta.problem_id,
                "problem_text": meta.problem_text,
                "expected_answer": meta.expected_answer,
                "filler_index": meta.filler_index,
                "filler": meta.filler,
                "response_text": None,
                "parsed_number": None,
                "is_correct": False,
                "error": str(raw),
            }
            records.append(record)
            continue

        message = raw.get("choices", [{}])[0].get("message", {})
        response_text = str(message.get("content", "")).strip()
        correct, parsed = is_correct(response_text, meta.expected_answer)
        record = {
            "request_index": meta.request_index,
            "problem_id": meta.problem_id,
            "problem_text": meta.problem_text,
            "expected_answer": meta.expected_answer,
            "filler_index": meta.filler_index,
            "filler": meta.filler,
            "response_text": response_text,
            "parsed_number": parsed,
            "is_correct": correct,
            "error": None,
            "raw_response": raw,
        }
        records.append(record)

    (run_dir / "results.json").write_text(json.dumps(records, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    by_filler: dict[str, dict[str, Any]] = {}
    for row in records:
        key = "CONTROL" if row["filler"] is None else f"FILLER_{row['filler_index']}"
        bucket = by_filler.setdefault(key, {"total": 0, "correct": 0, "errors": 0})
        bucket["total"] += 1
        if row["is_correct"]:
            bucket["correct"] += 1
        if row["error"] is not None:
            bucket["errors"] += 1

    summary_rows: list[dict[str, Any]] = []
    for key, stats in by_filler.items():
        total = stats["total"]
        correct = stats["correct"]
        summary_rows.append(
            {
                "condition": key,
                "total": total,
                "correct": correct,
                "accuracy": (correct / total) if total else 0.0,
                "errors": stats["errors"],
            }
        )
    summary_rows = sorted(summary_rows, key=lambda row: row["condition"])
    (run_dir / "summary.json").write_text(json.dumps(summary_rows, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    print(f"Run dir: {run_dir}")
    for row in summary_rows:
        print(
            f"{row['condition']}: accuracy={row['accuracy']:.3f} "
            f"({row['correct']}/{row['total']}), errors={row['errors']}"
        )


def count_main(args: argparse.Namespace) -> None:
    if args.problems_file:
        problems = load_problems(Path(args.problems_file))
    else:
        problems = generate_problems(
            count=args.problem_count,
            steps=args.steps,
            width=args.width,
            seed=args.problem_seed,
        )

    fillers = build_filler_set(
        variants=args.filler_variants,
        filler_length=args.filler_length,
        seed=args.filler_seed,
        include_control=not args.no_control,
        explicit_fillers=args.filler,
    )

    problem_count = len(problems)
    condition_count = len(fillers)
    total_inferences = problem_count * condition_count

    print(f"problems={problem_count}")
    print(f"conditions={condition_count}")
    print(f"total_inferences={total_inferences}")


def probe_main(args: argparse.Namespace) -> None:
    run_dir = ensure_output_run_dir(args.run_name)
    client = OpenRouterClient(
        OpenRouterConfig(
            model=args.model,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
        )
    )

    all_requests: list[dict[str, Any]] = []
    metas: list[dict[str, Any]] = []
    for steps in range(args.min_steps, args.max_steps + 1):
        for width in range(args.min_width, args.max_width + 1):
            problems = generate_problems(
                count=args.problems_per_bucket,
                steps=steps,
                width=width,
                seed=(args.seed + (steps * 1000) + (width * 10)),
            )
            for row in problems:
                prompt = compose_prompt(problem_text=row["text"], filler=None)
                all_requests.append({"user_prompt": prompt, "system_prompt": BASE_SYSTEM_PROMPT})
                metas.append(row)

    raw_results = client.chat_completion_batch(requests_batch=all_requests, max_workers=args.workers)
    detailed: list[dict[str, Any]] = []
    bucket_stats: dict[tuple[int, int], dict[str, int]] = {}
    for meta, raw in zip(metas, raw_results):
        key = (int(meta["steps"]), int(meta["width"]))
        stats = bucket_stats.setdefault(key, {"total": 0, "correct": 0, "errors": 0})
        stats["total"] += 1
        if isinstance(raw, Exception):
            stats["errors"] += 1
            detailed.append(
                {
                    "steps": meta["steps"],
                    "width": meta["width"],
                    "text": meta["text"],
                    "answer": meta["answer"],
                    "response_text": None,
                    "is_correct": False,
                    "error": str(raw),
                }
            )
            continue
        message = raw.get("choices", [{}])[0].get("message", {})
        response_text = str(message.get("content", "")).strip()
        correct, parsed = is_correct(response_text, meta["answer"])
        if correct:
            stats["correct"] += 1
        detailed.append(
            {
                "steps": meta["steps"],
                "width": meta["width"],
                "text": meta["text"],
                "answer": meta["answer"],
                "response_text": response_text,
                "parsed_number": parsed,
                "is_correct": correct,
                "error": None,
            }
        )

    summary: list[dict[str, Any]] = []
    for (steps, width), stats in sorted(bucket_stats.items()):
        total = stats["total"]
        correct = stats["correct"]
        summary.append(
            {
                "steps": steps,
                "width": width,
                "total": total,
                "correct": correct,
                "accuracy": (correct / total) if total else 0.0,
                "errors": stats["errors"],
            }
        )

    (run_dir / "probe_details.json").write_text(json.dumps(detailed, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    (run_dir / "probe_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    print(f"Run dir: {run_dir}")
    for row in summary:
        print(
            f"steps={row['steps']} width={row['width']}: "
            f"accuracy={row['accuracy']:.3f} ({row['correct']}/{row['total']}), errors={row['errors']}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filler-token arithmetic experiment runner.")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    def add_run_like_args(target: argparse.ArgumentParser) -> None:
        target.add_argument("--run-name", default="", help="Optional run folder name.")
        target.add_argument("--model", default="anthropic/claude-3.5-sonnet", help="OpenRouter model ID.")
        target.add_argument("--temperature", type=float, default=0.0)
        target.add_argument("--top-p", type=float, default=1.0)
        target.add_argument("--max-tokens", type=int, default=None)
        target.add_argument("--workers", type=int, default=8)

        target.add_argument("--problems-file", default="", help="JSON list (or {problems:[...]}) with text+answer.")
        target.add_argument("--problem-count", type=int, default=20, help="Used only when problems are generated.")
        target.add_argument("--steps", type=int, default=4, help="Operation count for generated problems.")
        target.add_argument("--width", type=int, default=2, help="Digit width for generated operands.")
        target.add_argument("--problem-seed", type=int, default=11)

        target.add_argument("--filler-variants", type=int, default=10, help="How many random filler strings.")
        target.add_argument("--filler-length", type=int, default=80)
        target.add_argument("--filler-seed", type=int, default=17)
        target.add_argument("--filler", action="append", help="Explicit filler string. Can be repeated.")
        target.add_argument("--no-control", action="store_true", help="Skip control condition without filler.")

    run_parser = subparsers.add_parser("run", help="Run filler-vs-control experiment.")
    add_run_like_args(run_parser)

    count_parser = subparsers.add_parser("count", help="Count planned inferences for run configuration.")
    add_run_like_args(count_parser)

    probe_parser = subparsers.add_parser("probe", help="Find difficulty buckets before full experiment.")
    probe_parser.add_argument("--run-name", default="", help="Optional run folder name.")
    probe_parser.add_argument("--model", default="anthropic/claude-3.5-sonnet", help="OpenRouter model ID.")
    probe_parser.add_argument("--temperature", type=float, default=0.0)
    probe_parser.add_argument("--top-p", type=float, default=1.0)
    probe_parser.add_argument("--max-tokens", type=int, default=None)
    probe_parser.add_argument("--workers", type=int, default=8)
    probe_parser.add_argument("--seed", type=int, default=101)
    probe_parser.add_argument("--problems-per-bucket", type=int, default=8)
    probe_parser.add_argument("--min-steps", type=int, default=3)
    probe_parser.add_argument("--max-steps", type=int, default=7)
    probe_parser.add_argument("--min-width", type=int, default=2)
    probe_parser.add_argument("--max-width", type=int, default=4)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "run":
        run_main(args)
        return
    if args.mode == "count":
        count_main(args)
        return
    probe_main(args)


if __name__ == "__main__":
    main()
