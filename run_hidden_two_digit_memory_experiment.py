"""Run a resumable experiment for hidden two-digit memory arithmetic."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from eval_constants import (
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    EVENTS_FILE_NAME,
    EXPERIMENT_OUTPUT_DIR,
    RUNS_DIR_NAME,
    STATE_FILE_NAME,
    STEPS_DIR_NAME,
)
from openrouter_client import OpenRouterClient, OpenRouterConfig


DEFAULT_EXPERIMENT_NAME = "hidden_two_digit_memory"
DEFAULT_TEST_MODEL = "anthropic/claude-sonnet-4.6"
DEFAULT_REASONING_EFFORT = "none"
DEFAULT_SECRET_TYPE = "number"
REDACTION_NOTICE = "that was the last thing you wrote before you gave your secret data, which we have now redacted..."


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_event(run_dir: Path, event_type: str, payload: dict[str, Any]) -> None:
    line = {"ts": now_iso(), "event": event_type, **payload}
    with (run_dir / EVENTS_FILE_NAME).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(line, ensure_ascii=True) + "\n")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    payload["updated_at"] = now_iso()
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def latest_run_dir(runs_dir: Path) -> Path | None:
    candidates = [p for p in runs_dir.iterdir() if p.is_dir()]
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: p.name)[-1]


def parse_secret_claim(text: str, secret_type: str) -> str | None:
    if secret_type == "number":
        match = re.search(r"my number is\s*(\d{2})", text, flags=re.IGNORECASE)
        return match.group(1) if match else None

    if secret_type == "word":
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if len(lines) < 2:
            return None
        secret_line = lines[-1]
        secret_match = re.fullmatch(r"[A-Za-z]+", secret_line)
        if not secret_match:
            return None
        secret_word = secret_match.group(0).lower()
        body = " ".join(lines[:-1])
        sentence_count = len(re.findall(r"[.!?]", body))
        if sentence_count < 2:
            return None
        body_words = [w.lower() for w in re.findall(r"[A-Za-z]+", body)]
        if secret_word in body_words:
            return None
        return secret_word

    if secret_type == "color":
        match = re.search(r"my color is\s*([A-Za-z]+(?:\s+[A-Za-z]+)?)", text, flags=re.IGNORECASE)
        return match.group(1).lower().strip() if match else None

    raise ValueError(f"Unsupported secret_type: {secret_type}")


def pre_secret_text(text: str, secret_type: str) -> str:
    if secret_type == "number":
        match = re.search(r"my number is\s*\d{2}", text, flags=re.IGNORECASE)
        return text[: match.start()].strip() if match else ""
    if secret_type == "word":
        lines = text.splitlines()
        if not lines:
            return ""
        return "\n".join(lines[:-1]).strip()
    if secret_type == "color":
        match = re.search(r"my color is\s*[A-Za-z]+(?:\s+[A-Za-z]+)?", text, flags=re.IGNORECASE)
        return text[: match.start()].strip() if match else ""
    raise ValueError(f"Unsupported secret_type: {secret_type}")


def init_or_resume_run(
    experiment_name: str,
    model: str,
    trials: int,
    reasoning_effort: str,
    secret_type: str,
    force_new_run: bool,
) -> tuple[Path, dict[str, Any]]:
    root = Path(EXPERIMENT_OUTPUT_DIR) / experiment_name
    runs_dir = root / RUNS_DIR_NAME
    runs_dir.mkdir(parents=True, exist_ok=True)

    if not force_new_run:
        latest = latest_run_dir(runs_dir)
        if latest is not None:
            state_path = latest / STATE_FILE_NAME
            if not state_path.exists():
                raise FileNotFoundError(f"Missing state file: {state_path}")
            state = read_json(state_path)
            if (
                state["model"] != model
                or int(state["trials"]) != int(trials)
                or state.get("reasoning_effort") != reasoning_effort
                or state.get("secret_type") != secret_type
            ):
                raise ValueError("Latest run config mismatch. Use --new-run or pass same config.")
            return latest, state

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = runs_dir / run_id
    (run_dir / STEPS_DIR_NAME).mkdir(parents=True, exist_ok=True)
    state = {
        "run_id": run_id,
        "experiment": experiment_name,
        "model": model,
        "trials": trials,
        "reasoning_effort": reasoning_effort,
        "secret_type": secret_type,
        "started_at": now_iso(),
        "updated_at": now_iso(),
        "status": "running",
        "completed_trials": 0,
        "successful_trials": 0,
    }
    write_json(run_dir / STATE_FILE_NAME, state)
    append_event(run_dir, "run_created", {"run_id": run_id})
    return run_dir, state


def run_experiment(
    experiment_name: str,
    model: str,
    trials: int,
    new_run: bool,
    temperature: float,
    top_p: float,
    max_tokens: int | None,
    reasoning_effort: str,
    secret_type: str,
) -> None:
    run_dir, state = init_or_resume_run(experiment_name, model, trials, reasoning_effort, secret_type, new_run)
    steps_dir = run_dir / STEPS_DIR_NAME
    steps_dir.mkdir(parents=True, exist_ok=True)

    completed = 0
    success_count = 0
    for step_file in sorted(steps_dir.glob("*.json")):
        try:
            row = read_json(step_file)
        except json.JSONDecodeError:
            continue
        if row.get("status") != "completed":
            continue
        completed += 1
        if row.get("evaluation", {}).get("success") is True:
            success_count += 1

    state["completed_trials"] = completed
    state["successful_trials"] = success_count
    write_json(run_dir / STATE_FILE_NAME, state)

    client = OpenRouterClient(
        OpenRouterConfig(
            model=model,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
    )

    if secret_type == "number":
        secret_label = "two-digit number"
        step_instruction = "Each step, output exactly: my number is <two-digit number>"
    elif secret_type == "word":
        secret_label = "single word"
        step_instruction = (
            "Each step, output exactly in this format:\n"
            "- First: two sentences thinking about the experiment so far.\n"
            "- Then: a newline.\n"
            "- Then: one random hidden word by itself on the last line.\n"
            "Do not use the hidden word anywhere in the two thinking sentences.\n"
            "Important: you must output the actual secret word on the last line. "
            "The evaluator will redact secret data afterward to test memory."
        )
    else:
        secret_label = "color name"
        step_instruction = "Each step, output exactly: my color is <color>"

    user_prompt = (
        "You are completing a controlled memory task.\n"
        f"Pick one hidden {secret_label} and keep it the same through this trial.\n"
        f"{step_instruction}\n"
        "Do not output any extra text."
    )

    append_event(run_dir, "run_started", {"run_id": state["run_id"]})

    for trial_index in range(1, trials + 1):
        step_file = steps_dir / f"{trial_index:04d}.json"
        if step_file.exists():
            continue

        append_event(run_dir, "trial_started", {"trial_index": trial_index})
        messages: list[dict[str, str]] = [{"role": "user", "content": user_prompt}]
        raw_turns: list[dict[str, Any]] = []
        trial_private_value: str | None = None
        all_steps_correct = True
        rounds = 4

        for n in range(1, rounds + 1):
            step_prompt = f"STEP {n}: output now."
            messages.append({"role": "user", "content": step_prompt})
            response_json = client.chat_completion_messages(
                messages=messages,
                extra_body={"reasoning": {"effort": reasoning_effort}},
            )
            message = response_json.get("choices", [{}])[0].get("message", {})
            content = (message.get("content", "") or "").strip()
            raw_turns.append({"kind": f"step_{n}", "prompt": step_prompt, "response": content, "raw_response": response_json})

            parsed_value = parse_secret_claim(content, secret_type)
            pre_text = pre_secret_text(content, secret_type).strip() or "(nothing before secret data)"

            if parsed_value is None:
                validator_msg = f"{REDACTION_NOTICE}\n{pre_text}\nvalidator: wrong"
                all_steps_correct = False
                messages.append({"role": "user", "content": validator_msg})
                raw_turns.append({"kind": f"validator_step_{n}", "response": validator_msg})
                continue

            if secret_type == "number":
                numeric = int(parsed_value)
                if not (10 <= numeric <= 99):
                    validator_msg = f"{REDACTION_NOTICE}\n{pre_text}\nvalidator: wrong"
                    all_steps_correct = False
                    messages.append({"role": "user", "content": validator_msg})
                    raw_turns.append({"kind": f"validator_step_{n}", "response": validator_msg})
                    continue

            if trial_private_value is None:
                trial_private_value = parsed_value

            step_correct = parsed_value == trial_private_value
            if not step_correct:
                all_steps_correct = False
                validator_msg = f"{REDACTION_NOTICE}\n{pre_text}\nvalidator: wrong"
            else:
                validator_msg = f"{REDACTION_NOTICE}\n{pre_text}\nvalidator: right"
            messages.append({"role": "user", "content": validator_msg})
            raw_turns.append(
                {
                    "kind": f"validator_step_{n}",
                    "response": validator_msg,
                    "parsed_value": parsed_value,
                    "pre_secret_text": pre_text,
                    "expected_value": trial_private_value,
                    "step_correct": step_correct,
                }
            )

        success = bool(all_steps_correct and trial_private_value is not None)
        evaluation = {
            "success": success,
            "all_steps_correct": all_steps_correct,
            "private_value": trial_private_value,
            "reason": "ok" if success else "One or more steps were wrong or unparsable.",
        }
        latest_model_raw = None
        for turn in reversed(raw_turns):
            if isinstance(turn, dict) and "raw_response" in turn:
                latest_model_raw = turn["raw_response"]
                break

        record = {
            "status": "completed",
            "completed_at": now_iso(),
            "trial_index": trial_index,
            "prompt": {"user_prompt": user_prompt, "reasoning_effort": reasoning_effort, "secret_type": secret_type},
            "model_output": raw_turns[-1]["response"] if raw_turns else "",
            "turns": raw_turns,
            "evaluation": evaluation,
            "raw_response": latest_model_raw,
        }
        write_json(step_file, record)
        append_event(
            run_dir,
            "trial_completed",
            {"trial_index": trial_index, "success": evaluation["success"], "step_file": str(step_file)},
        )

        completed += 1
        if evaluation["success"] is True:
            success_count += 1
        state["completed_trials"] = completed
        state["successful_trials"] = success_count
        write_json(run_dir / STATE_FILE_NAME, state)
        print(
            f"trial={trial_index}/{trials} success={evaluation['success']} "
            f"steps_ok={evaluation['all_steps_correct']}"
        )

    state["status"] = "completed"
    write_json(run_dir / STATE_FILE_NAME, state)
    append_event(run_dir, "run_completed", {"run_id": state["run_id"]})

    rate = (success_count / completed) if completed else 0.0
    print(f"Run complete: {run_dir}")
    print(f"Auto-scored success count: {success_count}/{completed} ({rate:.3f})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run hidden two-digit memory arithmetic experiment.")
    parser.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT_NAME)
    parser.add_argument("--model", default=DEFAULT_TEST_MODEL)
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--new-run", action="store_true")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--secret-type", default=DEFAULT_SECRET_TYPE, choices=["number", "word", "color"])
    parser.add_argument(
        "--reasoning-effort",
        default=DEFAULT_REASONING_EFFORT,
        choices=["xhigh", "high", "medium", "low", "minimal", "none"],
        help="OpenRouter reasoning.effort setting.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.trials < 1:
        raise ValueError("--trials must be >= 1")
    run_experiment(
        experiment_name=args.experiment_name,
        model=args.model,
        trials=args.trials,
        new_run=args.new_run,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        reasoning_effort=args.reasoning_effort,
        secret_type=args.secret_type,
    )


if __name__ == "__main__":
    main()
