"""Run a resumable math eval experiment against OpenRouter."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from eval_constants import (
    DEFAULT_MODEL,
    EVENTS_FILE_NAME,
    EXPERIMENT_NAME,
    EXPERIMENT_OUTPUT_DIR,
    REQUIRED_TASK_INSTRUCTION,
    RUNS_DIR_NAME,
    STATE_FILE_NAME,
    STYLE_INSTRUCTION_1,
    STYLE_INSTRUCTION_2,
    STEPS_DIR_NAME,
)
from openrouter_client import OpenRouterClient, OpenRouterConfig


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_questions(dataset_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a top-level object in {dataset_path}.")

    question_list = None
    for value in payload.values():
        if isinstance(value, list) and value and isinstance(value[0], dict) and "text" in value[0]:
            question_list = value
            break

    if question_list is None:
        raise ValueError(
            f"Could not find a question list in {dataset_path}. Expected a list of objects with a 'text' field."
        )

    normalized: list[dict[str, Any]] = []
    for idx, item in enumerate(question_list, start=1):
        normalized.append(
            {
                "index": idx,
                "number": item.get("number", idx),
                "text": item["text"],
                "final_answer": item.get("final_answer"),
            }
        )
    return normalized


def compose_prompt(question_text: str, style_instruction_1: str, style_instruction_2: str) -> str:
    parts = [REQUIRED_TASK_INSTRUCTION]
    if style_instruction_1.strip():
        parts.append(style_instruction_1.strip())
    parts.append(question_text.strip())
    if style_instruction_2.strip():
        parts.append(style_instruction_2.strip())
    return "\n\n".join(parts)


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


def init_or_resume_run(
    dataset_path: Path,
    total_questions: int,
    model: str,
    force_new_run: bool,
) -> tuple[Path, dict[str, Any]]:
    experiment_root = Path(EXPERIMENT_OUTPUT_DIR) / EXPERIMENT_NAME
    runs_dir = experiment_root / RUNS_DIR_NAME
    runs_dir.mkdir(parents=True, exist_ok=True)

    if not force_new_run:
        latest = latest_run_dir(runs_dir)
        if latest is not None:
            state_path = latest / STATE_FILE_NAME
            if not state_path.exists():
                raise FileNotFoundError(f"Missing state file in latest run: {state_path}")
            state = read_state(state_path)
            expected_dataset = Path(state["dataset_path"]).resolve()
            if expected_dataset != dataset_path.resolve():
                raise ValueError(
                    "Latest run uses a different dataset. Use --new-run or pass that same dataset path."
                )
            return latest, state

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = runs_dir / run_id
    (run_dir / STEPS_DIR_NAME).mkdir(parents=True, exist_ok=True)

    state = {
        "run_id": run_id,
        "experiment": EXPERIMENT_NAME,
        "dataset_path": str(dataset_path),
        "question_set": dataset_path.stem,
        "model": model,
        "started_at": now_iso(),
        "updated_at": now_iso(),
        "status": "running",
        "total_questions": total_questions,
        "completed_count": 0,
        "completed_numbers": [],
    }
    write_state(run_dir / STATE_FILE_NAME, state)
    append_event(run_dir, "run_created", {"run_id": run_id, "dataset_path": str(dataset_path)})
    return run_dir, state


def run_eval(
    dataset_path: Path,
    style_instruction_1: str,
    style_instruction_2: str,
    new_run: bool,
    model: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> None:
    questions = load_questions(dataset_path)
    question_set = dataset_path.stem
    run_dir, state = init_or_resume_run(
        dataset_path=dataset_path,
        total_questions=len(questions),
        model=model,
        force_new_run=new_run,
    )
    steps_dir = run_dir / STEPS_DIR_NAME
    steps_dir.mkdir(parents=True, exist_ok=True)

    # Rebuild completion state from on-disk step files so resume is robust.
    completed_numbers: set[int] = set()
    for step_file in sorted(steps_dir.glob("*.json")):
        try:
            step = json.loads(step_file.read_text(encoding="utf-8"))
            if step.get("status") == "completed":
                number = step.get("question", {}).get("number")
                if isinstance(number, int):
                    completed_numbers.add(number)
        except json.JSONDecodeError:
            append_event(run_dir, "warning", {"message": f"Invalid JSON in step file: {step_file}"})

    state["completed_numbers"] = sorted(completed_numbers)
    state["completed_count"] = len(completed_numbers)
    write_state(run_dir / STATE_FILE_NAME, state)

    client = OpenRouterClient(
        OpenRouterConfig(
            model=model,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
    )

    append_event(run_dir, "run_started", {"run_id": state["run_id"]})

    for question in questions:
        step_file = steps_dir / f"{question['index']:04d}_q{question['number']}.json"
        if step_file.exists():
            continue

        prompt = compose_prompt(question["text"], style_instruction_1, style_instruction_2)
        append_event(
            run_dir,
            "question_started",
            {
                "question_index": question["index"],
                "question_number": question["number"],
            },
        )

        response_json = client.chat_completion(user_prompt=prompt)
        message = response_json.get("choices", [{}])[0].get("message", {})
        content = message.get("content", "").strip()
        reasoning = message.get("reasoning")

        record = {
            "status": "completed",
            "completed_at": now_iso(),
            "question_set": question_set,
            "question": question,
            "prompt": {
                "required_instruction": REQUIRED_TASK_INSTRUCTION,
                "style_instruction_1": style_instruction_1,
                "question_text": question["text"],
                "style_instruction_2": style_instruction_2,
                "full_prompt": prompt,
            },
            "model_output": content,
            "model_reasoning": reasoning,
            "score": None,
            "raw_response": response_json,
        }
        step_file.write_text(json.dumps(record, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

        print(f"Q{question['number']} OUTPUT: {content}")
        print(f"Q{question['number']} REASONING: {reasoning}")

        completed_numbers = set(state.get("completed_numbers", []))
        completed_numbers.add(question["number"])
        state["completed_numbers"] = sorted(completed_numbers)
        state["completed_count"] = len(state["completed_numbers"])
        write_state(run_dir / STATE_FILE_NAME, state)

        append_event(
            run_dir,
            "question_completed",
            {
                "question_index": question["index"],
                "question_number": question["number"],
                "step_file": str(step_file),
            },
        )

    state["status"] = "completed"
    write_state(run_dir / STATE_FILE_NAME, state)
    append_event(run_dir, "run_completed", {"run_id": state["run_id"]})

    print(f"Run complete: {run_dir}")
    print(f"Completed: {state['completed_count']} / {state['total_questions']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a resumable OpenRouter math eval.")
    parser.add_argument("--dataset", default="data/sprint.json", help="Path to dataset JSON file.")
    parser.add_argument(
        "--style-instruction-1",
        "--optional-part-1",
        dest="style_instruction_1",
        default=STYLE_INSTRUCTION_1,
        help="Style instruction inserted before the question.",
    )
    parser.add_argument(
        "--style-instruction-2",
        "--optional-part-2",
        dest="style_instruction_2",
        default=STYLE_INSTRUCTION_2,
        help="Style instruction inserted after the question.",
    )
    parser.add_argument("--new-run", action="store_true", help="Start a new run instead of resuming latest.")

    parser.add_argument("--model", default=DEFAULT_MODEL, help="OpenRouter model id.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_eval(
        dataset_path=Path(args.dataset),
        style_instruction_1=args.style_instruction_1,
        style_instruction_2=args.style_instruction_2,
        new_run=args.new_run,
        model=args.model,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )


if __name__ == "__main__":
    main()
