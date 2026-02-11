"""Score a run with answer correctness and an OpenRouter style-adherence judge."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any

from eval_constants import (
    DEFAULT_JUDGE_MODEL,
    EXPERIMENT_NAME,
    EXPERIMENT_OUTPUT_DIR,
    JUDGE_RERUN_LIMIT,
    JUDGE_SYSTEM_PROMPT,
    RUNS_DIR_NAME,
    STEPS_DIR_NAME,
)
from openrouter_client import OpenRouterClient, OpenRouterConfig


def latest_run_dir(runs_dir: Path) -> Path | None:
    candidates = [p for p in runs_dir.iterdir() if p.is_dir()]
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: p.name)[-1]


def normalize_text_answer(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value).strip().lower().replace(" ", "")
    return text


def try_parse_float(value: Any) -> float | None:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def is_answer_correct(predicted: Any, expected: Any) -> bool:
    predicted_num = try_parse_float(predicted)
    expected_num = try_parse_float(expected)
    if predicted_num is not None and expected_num is not None:
        return abs(predicted_num - expected_num) <= 1e-6
    return normalize_text_answer(predicted) == normalize_text_answer(expected)


def parse_single_score(text: str) -> int | None:
    cleaned = text.strip()
    if re.fullmatch(r"[0-5]", cleaned):
        return int(cleaned)
    return None


def style_flags_from_score(score: int | None) -> tuple[bool | None, bool | None]:
    if score is None:
        return None, None
    acknowledged = score in {1, 2, 4}
    successful = score in {4, 5}
    return acknowledged, successful


def judge_style_adherence(
    client: OpenRouterClient,
    judge_model: str,
    question_text: str,
    expected_answer: Any,
    model_answer: Any,
    model_reasoning: Any,
    style_instruction_1: str,
    style_instruction_2: str,
) -> dict[str, Any]:
    base_prompt = (
        "Evaluate whether the model adhered to the style instructions.\\n\\n"
        f"Style instruction 1: {style_instruction_1}\\n"
        f"Style instruction 2: {style_instruction_2}\\n\\n"
        f"Question: {question_text}\\n"
        f"Ground truth final answer: {expected_answer}\\n"
        f"Model final answer: {model_answer}\\n"
        f"Model reasoning summary: {model_reasoning}\\n\\n"
        "Use the 0-5 rubric exactly as instructed in the system prompt.\\n"
        "Return exactly one character: 0, 1, 2, 3, 4, or 5."
    )

    attempts: list[dict[str, Any]] = []
    score: int | None = None
    for attempt in range(1, JUDGE_RERUN_LIMIT + 2):
        user_prompt = base_prompt
        if attempt > 1:
            user_prompt += "\\nPrevious output was invalid. Output one digit only: 0-5."

        response = client.chat_completion(user_prompt=user_prompt, system_prompt=JUDGE_SYSTEM_PROMPT)
        message = response.get("choices", [{}])[0].get("message", {})
        content = message.get("content", "")
        score = parse_single_score(content)
        attempts.append({"attempt": attempt, "judge_text": content, "judge_raw": response})
        if score is not None:
            break

    style_acknowledged, style_adherence_successful = style_flags_from_score(score)
    final_attempt = attempts[-1]

    return {
        "judge_model": judge_model,
        "judge_attempt_count": len(attempts),
        "judge_rerun_limit": JUDGE_RERUN_LIMIT,
        "judge_raw": final_attempt["judge_raw"],
        "judge_text": final_attempt["judge_text"],
        "judge_attempts": attempts,
        "style_score_0_to_5": score,
        "style_acknowledged": style_acknowledged,
        "style_adherence_successful": style_adherence_successful,
    }


def to_markdown_table(rows: list[dict[str, Any]]) -> str:
    headers = [
        "question_set",
        "question_number",
        "question_correct",
        "answer_score",
        "style_score_0_to_5",
        "style_acknowledged",
        "style_adherence_successful",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        values = [str(row.get(col, "")) for col in headers]
        lines.append("| " + " | ".join(values) + " |")
    return "\\n".join(lines)


def score_run(run_dir: Path, judge_model: str, no_judge: bool) -> None:
    state_path = run_dir / "state.json"
    if not state_path.exists():
        raise FileNotFoundError(f"Missing state file: {state_path}")

    state = json.loads(state_path.read_text(encoding="utf-8"))
    dataset_path = Path(state.get("dataset_path", ""))
    question_set = state.get("question_set") or dataset_path.stem or "unknown"

    step_files = sorted((run_dir / STEPS_DIR_NAME).glob("*.json"))
    if not step_files:
        raise FileNotFoundError(f"No step files found in: {run_dir / STEPS_DIR_NAME}")

    judge_client = None
    if not no_judge:
        judge_client = OpenRouterClient(OpenRouterConfig(model=judge_model, temperature=0.0, max_tokens=300))

    rows: list[dict[str, Any]] = []
    for step_file in step_files:
        step = json.loads(step_file.read_text(encoding="utf-8"))
        question = step.get("question", {})
        prompt = step.get("prompt", {})
        style_instruction_1 = prompt.get("style_instruction_1", prompt.get("optional_part_1", ""))
        style_instruction_2 = prompt.get("style_instruction_2", prompt.get("optional_part_2", ""))
        predicted = step.get("model_output")
        expected = question.get("final_answer")
        correct = is_answer_correct(predicted, expected)

        row: dict[str, Any] = {
            "question_set": question_set,
            "question_number": question.get("number"),
            "question_correct": correct,
            "answer_score": 1 if correct else 0,
            "style_score_0_to_5": None,
            "style_acknowledged": None,
            "style_adherence_successful": None,
        }

        if judge_client is not None:
            judged = judge_style_adherence(
                client=judge_client,
                judge_model=judge_model,
                question_text=question.get("text", ""),
                expected_answer=expected,
                model_answer=predicted,
                model_reasoning=step.get("model_reasoning"),
                style_instruction_1=style_instruction_1,
                style_instruction_2=style_instruction_2,
            )
            row.update(
                {
                    "style_score_0_to_5": judged["style_score_0_to_5"],
                    "style_acknowledged": judged["style_acknowledged"],
                    "style_adherence_successful": judged["style_adherence_successful"],
                }
            )
            step["score"] = {
                "answer_correct": correct,
                "answer_score": row["answer_score"],
                "style_judge": judged,
            }
        else:
            step["score"] = {
                "answer_correct": correct,
                "answer_score": row["answer_score"],
                "style_judge": None,
            }

        step_file.write_text(json.dumps(step, indent=2, ensure_ascii=True) + "\\n", encoding="utf-8")
        rows.append(row)
        print(
            f"Scored Q{row['question_number']}: correct={row['question_correct']} "
            f"style_score={row['style_score_0_to_5']}"
        )

    scores_json = run_dir / "scores.json"
    scores_md = run_dir / "scores.md"
    markdown_table = to_markdown_table(rows)

    scores_json.write_text(json.dumps(rows, indent=2, ensure_ascii=True) + "\\n", encoding="utf-8")
    scores_md.write_text(markdown_table + "\\n", encoding="utf-8")

    print(f"Wrote: {scores_json}")
    print(f"Wrote: {scores_md}")
    print(markdown_table)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score a math-eval run.")
    parser.add_argument(
        "--run-dir",
        default="",
        help="Run directory to score. If omitted, scores the latest run.",
    )
    parser.add_argument(
        "--judge-model",
        default=DEFAULT_JUDGE_MODEL,
        help="OpenRouter model for style-adherence judge.",
    )
    parser.add_argument(
        "--no-judge",
        action="store_true",
        help="Skip LLM judge and only compute answer correctness.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        runs_dir = Path(EXPERIMENT_OUTPUT_DIR) / EXPERIMENT_NAME / RUNS_DIR_NAME
        if not runs_dir.exists():
            raise FileNotFoundError(f"No runs directory found: {runs_dir}")
        latest = latest_run_dir(runs_dir)
        if latest is None:
            raise FileNotFoundError(f"No runs found in: {runs_dir}")
        run_dir = latest

    score_run(run_dir=run_dir, judge_model=args.judge_model, no_judge=args.no_judge)


if __name__ == "__main__":
    main()
