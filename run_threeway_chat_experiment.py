"""Run a resumable three-way chat experiment with two LLMs and one admin user."""

from __future__ import annotations

import argparse
import json
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


DEFAULT_EXPERIMENT_NAME = "threeway_chat"
DEFAULT_TURN_ORDER = ["admin", "model_a", "model_b"]
VALID_SPEAKERS = {"admin", "model_a", "model_b"}
SPEAKER_LABELS = {
    "admin": "ADMIN",
    "model_a": "MODEL_A",
    "model_b": "MODEL_B",
}


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


def parse_turn_order(raw_turn_order: str) -> list[str]:
    order = [token.strip() for token in raw_turn_order.split(",") if token.strip()]
    if not order:
        raise ValueError("Turn order cannot be empty.")
    for speaker in order:
        if speaker not in VALID_SPEAKERS:
            raise ValueError(f"Invalid speaker '{speaker}' in turn order. Valid values: {sorted(VALID_SPEAKERS)}")
    return order


def load_admin_messages(admin_script_path: Path) -> list[str]:
    payload = json.loads(admin_script_path.read_text(encoding="utf-8"))
    messages: list[str]

    if isinstance(payload, list):
        messages = payload
    elif isinstance(payload, dict) and isinstance(payload.get("messages"), list):
        messages = payload["messages"]
    else:
        raise ValueError(
            f"Expected {admin_script_path} to be either a JSON list of strings "
            "or an object with a 'messages' list."
        )

    normalized: list[str] = []
    for idx, item in enumerate(messages, start=1):
        if not isinstance(item, str):
            raise ValueError(f"Admin message at index {idx} must be a string.")
        cleaned = item.strip()
        if not cleaned:
            raise ValueError(f"Admin message at index {idx} is empty.")
        normalized.append(cleaned)
    return normalized


def build_history_transcript(history: list[dict[str, Any]]) -> str:
    if not history:
        return "(No prior turns yet.)"

    lines: list[str] = []
    for entry in history:
        speaker = entry["speaker"]
        label = SPEAKER_LABELS.get(speaker, speaker.upper())
        lines.append(f"{label}: {entry['text']}")
    return "\n\n".join(lines)


def stringify_reasoning(raw_reasoning: Any) -> str:
    if raw_reasoning is None:
        return ""
    if isinstance(raw_reasoning, str):
        return raw_reasoning.strip()
    return json.dumps(raw_reasoning, ensure_ascii=True)


def ensure_think_prefix(content: str, raw_reasoning: Any) -> str:
    cleaned_content = content.strip()
    if cleaned_content.startswith("<think>"):
        return cleaned_content

    reasoning = stringify_reasoning(raw_reasoning)
    if not reasoning:
        reasoning = "No separate reasoning field returned by API."
    body = cleaned_content if cleaned_content else "(No response text returned.)"
    return f"<think>\n{reasoning}\n</think>\n{body}"


def build_model_prompts(
    speaker: str,
    history: list[dict[str, Any]],
    model_a_system: str,
    model_b_system: str,
) -> tuple[str, str]:
    speaker_label = SPEAKER_LABELS[speaker]
    other_a = SPEAKER_LABELS["model_a"]
    other_b = SPEAKER_LABELS["model_b"]

    custom_system = model_a_system if speaker == "model_a" else model_b_system
    system_prompt_parts = [
        f"You are {speaker_label} in a three-way conversation with ADMIN, {other_a}, and {other_b}.",
        "Only speak as yourself. Do not produce messages for other participants.",
        "Your response must include your reasoning first enclosed in <think>...</think>, then the visible reply.",
    ]
    if custom_system.strip():
        system_prompt_parts.append(custom_system.strip())
    system_prompt = "\n".join(system_prompt_parts)

    transcript = build_history_transcript(history)
    user_prompt = (
        f"Conversation transcript so far:\n\n{transcript}\n\n"
        f"Now produce the next turn as {speaker_label}. "
        "Remember: include your reasoning in <think> tags before your visible response."
    )
    return system_prompt, user_prompt


def rebuild_history_from_steps(steps_dir: Path) -> list[dict[str, Any]]:
    history: list[dict[str, Any]] = []
    for step_file in sorted(steps_dir.glob("*.json")):
        try:
            step = json.loads(step_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if step.get("status") != "completed":
            continue
        history_entry = step.get("history_entry")
        if isinstance(history_entry, dict):
            history.append(history_entry)
    return history


def init_or_resume_run(
    experiment_name: str,
    admin_script_path: Path,
    model_a: str,
    model_b: str,
    turn_order: list[str],
    max_turns: int,
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
            if Path(state["admin_script_path"]).resolve() != admin_script_path.resolve():
                raise ValueError("Latest run uses a different admin script. Use --new-run or pass the same script.")
            if state["model_a"] != model_a or state["model_b"] != model_b:
                raise ValueError("Latest run uses different model IDs. Use --new-run or pass the same models.")
            if state["turn_order"] != turn_order:
                raise ValueError("Latest run uses a different turn order. Use --new-run or pass the same order.")
            if int(state["max_turns"]) != int(max_turns):
                raise ValueError("Latest run uses a different max-turns. Use --new-run or pass the same value.")
            return latest, state

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = runs_dir / run_id
    (run_dir / STEPS_DIR_NAME).mkdir(parents=True, exist_ok=True)

    state = {
        "run_id": run_id,
        "experiment": experiment_name,
        "admin_script_path": str(admin_script_path),
        "model_a": model_a,
        "model_b": model_b,
        "turn_order": turn_order,
        "max_turns": max_turns,
        "started_at": now_iso(),
        "updated_at": now_iso(),
        "status": "running",
        "completed_turns": 0,
        "admin_messages_used": 0,
    }
    write_state(run_dir / STATE_FILE_NAME, state)
    append_event(run_dir, "run_created", {"run_id": run_id, "admin_script_path": str(admin_script_path)})
    return run_dir, state


def run_threeway_chat(
    experiment_name: str,
    admin_script_path: Path,
    model_a: str,
    model_b: str,
    model_a_system: str,
    model_b_system: str,
    turn_order: list[str],
    max_turns: int,
    new_run: bool,
    temperature: float,
    top_p: float,
    max_tokens: int | None,
) -> None:
    admin_messages = load_admin_messages(admin_script_path)
    run_dir, state = init_or_resume_run(
        experiment_name=experiment_name,
        admin_script_path=admin_script_path,
        model_a=model_a,
        model_b=model_b,
        turn_order=turn_order,
        max_turns=max_turns,
        force_new_run=new_run,
    )

    steps_dir = run_dir / STEPS_DIR_NAME
    steps_dir.mkdir(parents=True, exist_ok=True)

    history = rebuild_history_from_steps(steps_dir)
    completed_turns = len(history)
    admin_messages_used = sum(1 for h in history if h.get("speaker") == "admin")
    state["completed_turns"] = completed_turns
    state["admin_messages_used"] = admin_messages_used
    write_state(run_dir / STATE_FILE_NAME, state)

    client_a = OpenRouterClient(
        OpenRouterConfig(
            model=model_a,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
    )
    client_b = OpenRouterClient(
        OpenRouterConfig(
            model=model_b,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
    )

    append_event(run_dir, "run_started", {"run_id": state["run_id"]})

    while completed_turns < max_turns:
        speaker = turn_order[completed_turns % len(turn_order)]
        step_file = steps_dir / f"{completed_turns + 1:04d}_{speaker}.json"
        if step_file.exists():
            completed_turns += 1
            continue

        append_event(
            run_dir,
            "turn_started",
            {"turn_index": completed_turns + 1, "speaker": speaker},
        )

        if speaker == "admin":
            if admin_messages_used >= len(admin_messages):
                append_event(
                    run_dir,
                    "run_stopped",
                    {
                        "reason": "admin_messages_exhausted",
                        "completed_turns": completed_turns,
                        "admin_messages_used": admin_messages_used,
                    },
                )
                state["status"] = "completed"
                write_state(run_dir / STATE_FILE_NAME, state)
                print(f"Run complete: {run_dir}")
                print(f"Stopped because admin messages were exhausted at turn {completed_turns + 1}.")
                print(f"Completed turns: {completed_turns} / {max_turns}")
                return

            admin_text = admin_messages[admin_messages_used]
            history_entry = {
                "turn_index": completed_turns + 1,
                "speaker": "admin",
                "text": admin_text,
                "created_at": now_iso(),
            }
            record = {
                "status": "completed",
                "completed_at": now_iso(),
                "turn_index": completed_turns + 1,
                "speaker": "admin",
                "admin_message_index": admin_messages_used,
                "history_entry": history_entry,
            }
            step_file.write_text(json.dumps(record, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
            history.append(history_entry)
            admin_messages_used += 1
            print(f"TURN {completed_turns + 1} ADMIN: {admin_text}")
        else:
            system_prompt, user_prompt = build_model_prompts(
                speaker=speaker,
                history=history,
                model_a_system=model_a_system,
                model_b_system=model_b_system,
            )
            client = client_a if speaker == "model_a" else client_b
            response_json = client.chat_completion(user_prompt=user_prompt, system_prompt=system_prompt)
            message = response_json.get("choices", [{}])[0].get("message", {})
            content = message.get("content", "")
            reasoning = message.get("reasoning")
            combined_text = ensure_think_prefix(content, reasoning)

            history_entry = {
                "turn_index": completed_turns + 1,
                "speaker": speaker,
                "text": combined_text,
                "created_at": now_iso(),
            }
            record = {
                "status": "completed",
                "completed_at": now_iso(),
                "turn_index": completed_turns + 1,
                "speaker": speaker,
                "model": model_a if speaker == "model_a" else model_b,
                "prompt": {
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                },
                "model_output": combined_text,
                "raw_content": content,
                "raw_reasoning": reasoning,
                "raw_response": response_json,
                "history_entry": history_entry,
            }
            step_file.write_text(json.dumps(record, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
            history.append(history_entry)
            print(f"TURN {completed_turns + 1} {SPEAKER_LABELS[speaker]}: {combined_text}")

        append_event(
            run_dir,
            "turn_completed",
            {"turn_index": completed_turns + 1, "speaker": speaker, "step_file": str(step_file)},
        )

        completed_turns += 1
        state["completed_turns"] = completed_turns
        state["admin_messages_used"] = admin_messages_used
        write_state(run_dir / STATE_FILE_NAME, state)

    state["status"] = "completed"
    write_state(run_dir / STATE_FILE_NAME, state)
    append_event(run_dir, "run_completed", {"run_id": state["run_id"]})
    print(f"Run complete: {run_dir}")
    print(f"Completed turns: {completed_turns} / {max_turns}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a resumable three-way chat with two OpenRouter models.")
    parser.add_argument("--admin-script", required=True, help="Path to JSON admin script (list or {messages:[...]}).")
    parser.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT_NAME, help="Experiment output namespace.")
    parser.add_argument("--new-run", action="store_true", help="Start a new run instead of resuming latest.")

    parser.add_argument("--model-a", default=DEFAULT_MODEL, help="OpenRouter model ID for model_a.")
    parser.add_argument("--model-b", default=DEFAULT_MODEL, help="OpenRouter model ID for model_b.")
    parser.add_argument("--model-a-system", default="", help="Extra system prompt text for model_a.")
    parser.add_argument("--model-b-system", default="", help="Extra system prompt text for model_b.")

    parser.add_argument(
        "--turn-order",
        default=",".join(DEFAULT_TURN_ORDER),
        help="Comma-separated speaker order using: admin,model_a,model_b",
    )
    parser.add_argument("--max-turns", type=int, default=30, help="Total number of turns across all speakers.")

    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    parser.add_argument("--max-tokens", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    turn_order = parse_turn_order(args.turn_order)
    run_threeway_chat(
        experiment_name=args.experiment_name,
        admin_script_path=Path(args.admin_script),
        model_a=args.model_a,
        model_b=args.model_b,
        model_a_system=args.model_a_system,
        model_b_system=args.model_b_system,
        turn_order=turn_order,
        max_turns=args.max_turns,
        new_run=args.new_run,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )


if __name__ == "__main__":
    main()
