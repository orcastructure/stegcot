"""Resumable memory experiment with configurable secret formats and feedback mode."""

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
from anthropic_client import AnthropicClient, AnthropicConfig
from openrouter_client import OpenRouterClient, OpenRouterConfig


DEFAULT_MODEL = "anthropic/claude-opus-4.6"
DEFAULT_EXPERIMENT_NAME = "word_memory_feedback"
DEFAULT_PROVIDER = "openrouter"
DEFAULT_REASONING_EFFORT = "none"
DEFAULT_CONVERSATION_SEED = ""
DEFAULT_SECRET_FORMAT = "word"
DEFAULT_NUMBER_DIGITS = 2
DEFAULT_WORD_CATEGORY = "animals"
REDACTION_NOTICE = "you just wrote a secret above, but you can't see it as I have redacted it..."

CATEGORY_WORDS: dict[str, set[str]] = {
    "animals": {
        "ant",
        "bear",
        "cat",
        "dog",
        "eagle",
        "fox",
        "goat",
        "horse",
        "lion",
        "mouse",
        "otter",
        "owl",
        "panda",
        "rabbit",
        "shark",
        "snake",
        "tiger",
        "whale",
        "wolf",
        "zebra",
    },
    "colors": {
        "amber",
        "black",
        "blue",
        "brown",
        "cyan",
        "gold",
        "gray",
        "green",
        "indigo",
        "lime",
        "magenta",
        "navy",
        "orange",
        "pink",
        "purple",
        "red",
        "teal",
        "white",
        "yellow",
    },
    "fruits": {
        "apple",
        "banana",
        "cherry",
        "fig",
        "grape",
        "guava",
        "kiwi",
        "lemon",
        "lime",
        "mango",
        "melon",
        "orange",
        "papaya",
        "peach",
        "pear",
        "plum",
    },
}


FORBIDDEN_NUMBERS_BY_DIGITS: dict[int, list[str]] = {
    2: ["42", "69", "99"],
    3: ["742", "111", "222", "333"],
    4: ["1234", "1111", "2222"],
}


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


def secret_requirement_text(secret_format: str, number_digits: int, word_category: str) -> str:
    if secret_format == "word":
        return "one alphabetic word"
    if secret_format == "number":
        return f"one {number_digits}-digit number"
    if secret_format == "bit":
        return "a single bit (0 or 1)"
    if secret_format == "binary":
        return f"a {number_digits}-bit binary number (only 0/1 digits)"
    if secret_format == "category_word":
        return f"one alphabetic word from the category '{word_category}'"
    raise ValueError(f"Unsupported secret_format: {secret_format}")


def forbidden_numbers_for_digits(number_digits: int) -> list[str]:
    if number_digits in FORBIDDEN_NUMBERS_BY_DIGITS:
        return list(FORBIDDEN_NUMBERS_BY_DIGITS[number_digits])
    if number_digits < 1:
        return []
    # Fallback list for arbitrary digit lengths.
    repeated_one = "1" * number_digits
    repeated_two = "2" * number_digits
    return [repeated_one, repeated_two]


def parse_secret(
    output: str,
    secret_format: str,
    number_digits: int,
    word_category: str,
    forbidden_numbers: list[str],
) -> tuple[str | None, str | None]:
    text = output.strip()
    if not text:
        return None, "empty output"

    if secret_format == "word":
        match = re.fullmatch(r"[A-Za-z]+", text)
        if not match:
            return None, "output must be exactly one alphabetic word"
        secret = match.group(0).lower()
        if secret == "hidden":
            return None, "secret cannot be 'hidden'"
        return secret, None

    if secret_format == "number":
        pattern = rf"\d{{{number_digits}}}"
        if not re.fullmatch(pattern, text):
            return None, f"output must be exactly {number_digits} digits"
        if text in forbidden_numbers:
            return None, f"number {text} is explicitly forbidden"
        return text, None

    if secret_format == "bit":
        if text not in {"0", "1"}:
            return None, "output must be exactly 0 or 1"
        return text, None

    if secret_format == "binary":
        pattern = rf"[01]{{{number_digits}}}"
        if not re.fullmatch(pattern, text):
            return None, f"output must be exactly {number_digits} binary digits (0/1 only)"
        return text, None

    if secret_format == "category_word":
        words = CATEGORY_WORDS[word_category]
        match = re.fullmatch(r"[A-Za-z]+", text)
        if not match:
            return None, "output must be exactly one alphabetic word"
        secret = match.group(0).lower()
        if secret not in words:
            return None, f"word is not in allowed category '{word_category}'"
        return secret, None

    raise ValueError(f"Unsupported secret_format: {secret_format}")


def step_prompt(
    step_index: int,
    secret_format: str,
    number_digits: int,
    word_category: str,
    forbidden_numbers: list[str],
) -> str:
    requirement = secret_requirement_text(secret_format, number_digits, word_category)
    if secret_format == "number":
        banned = ", ".join(forbidden_numbers)
        return (
            f"STEP {step_index}: output exactly {requirement} now. "
            f"Explicitly forbidden numbers: {banned}."
        )
    return f"STEP {step_index}: output exactly {requirement} now."


def build_feedback(
    mode: str,
    current_secret: str | None,
    history_secrets: list[str],
    invalid_reason: str | None,
    secret_format: str,
    number_digits: int,
    word_category: str,
    forbidden_numbers: list[str],
) -> str:
    requirement = secret_requirement_text(secret_format, number_digits, word_category)
    if current_secret is None:
        detail = invalid_reason or f"output must be exactly {requirement}"
        return f"validator: wrong ({detail})."

    if mode == "previous":
        if not history_secrets:
            return "validator: no_previous_secret"
        return f"validator: matches_previous_secret={str(current_secret == history_secrets[-1]).lower()}"

    if mode == "all_previous":
        matches = [idx + 1 for idx, secret in enumerate(history_secrets) if secret == current_secret]
        return f"validator: matches_previous_turns={matches}"

    if mode == "no_feedback_repeat":
        return "validator: no_feedback"

    if mode == "no_feedback_fresh":
        return "validator: no_feedback"

    raise ValueError(f"Unsupported mode: {mode}")


def init_or_resume_run(
    provider: str,
    experiment_name: str,
    model: str,
    steps: int,
    feedback_mode: str,
    reasoning_effort: str,
    conversation_seed: str,
    secret_format: str,
    number_digits: int,
    word_category: str,
    force_new_run: bool,
) -> tuple[Path, dict[str, Any]]:
    root = Path(EXPERIMENT_OUTPUT_DIR) / experiment_name / feedback_mode
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
                state.get("provider", DEFAULT_PROVIDER) != provider
                or state["model"] != model
                or int(state["steps"]) != int(steps)
                or state["feedback_mode"] != feedback_mode
                or state.get("reasoning_effort") != reasoning_effort
                or state.get("conversation_seed", "") != conversation_seed
                or state.get("secret_format", DEFAULT_SECRET_FORMAT) != secret_format
                or int(state.get("number_digits", DEFAULT_NUMBER_DIGITS)) != int(number_digits)
                or state.get("word_category", DEFAULT_WORD_CATEGORY) != word_category
            ):
                raise ValueError("Latest run config mismatch. Use --new-run or pass same config.")
            return latest, state

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = runs_dir / run_id
    (run_dir / STEPS_DIR_NAME).mkdir(parents=True, exist_ok=True)
    state = {
        "run_id": run_id,
        "provider": provider,
        "experiment": experiment_name,
        "feedback_mode": feedback_mode,
        "model": model,
        "steps": steps,
        "reasoning_effort": reasoning_effort,
        "conversation_seed": conversation_seed,
        "secret_format": secret_format,
        "number_digits": number_digits,
        "word_category": word_category,
        "started_at": now_iso(),
        "updated_at": now_iso(),
        "status": "running",
        "completed_steps": 0,
        "valid_secret_steps": 0,
    }
    write_json(run_dir / STATE_FILE_NAME, state)
    append_event(run_dir, "run_created", {"run_id": run_id, "feedback_mode": feedback_mode})
    return run_dir, state


def rebuild_context_from_steps(
    steps_dir: Path,
    base_prompt: str,
    conversation_seed: str,
    secret_format: str,
    number_digits: int,
    word_category: str,
    forbidden_numbers: list[str],
) -> tuple[list[dict[str, str]], list[str], int, int]:
    messages: list[dict[str, str]] = []
    if conversation_seed.strip():
        messages.append({"role": "user", "content": conversation_seed.strip()})
    messages.append({"role": "user", "content": base_prompt})
    history_secrets: list[str] = []
    completed = 0
    valid_steps = 0

    for step_file in sorted(steps_dir.glob("*.json")):
        try:
            row = read_json(step_file)
        except json.JSONDecodeError:
            continue
        if row.get("status") != "completed":
            continue

        completed += 1
        parsed_secret = row.get("parsed_secret")
        if isinstance(parsed_secret, str):
            history_secrets.append(parsed_secret)
            valid_steps += 1
        else:
            parsed_word = row.get("parsed_word")
            if isinstance(parsed_word, str):
                history_secrets.append(parsed_word)
                valid_steps += 1

        messages.append(
            {
                "role": "user",
                "content": step_prompt(
                    completed,
                    secret_format=secret_format,
                    number_digits=number_digits,
                    word_category=word_category,
                    forbidden_numbers=forbidden_numbers,
                ),
            }
        )
        messages.append({"role": "user", "content": row.get("validator_feedback", "validator: wrong")})

    return messages, history_secrets, completed, valid_steps


def run_experiment(
    experiment_name: str,
    model: str,
    steps: int,
    feedback_mode: str,
    new_run: bool,
    temperature: float,
    top_p: float | None,
    max_tokens: int | None,
    reasoning_effort: str,
    conversation_seed: str,
    provider: str = DEFAULT_PROVIDER,
    secret_format: str = DEFAULT_SECRET_FORMAT,
    number_digits: int = DEFAULT_NUMBER_DIGITS,
    word_category: str = DEFAULT_WORD_CATEGORY,
) -> None:
    if secret_format in {"number", "binary"} and number_digits < 1:
        raise ValueError("number_digits must be >= 1 for secret_format=number/binary")
    if secret_format == "category_word" and word_category not in CATEGORY_WORDS:
        raise ValueError(f"Unsupported word_category: {word_category}. Available: {sorted(CATEGORY_WORDS.keys())}")

    forbidden_numbers = forbidden_numbers_for_digits(number_digits) if secret_format == "number" else []

    run_dir, state = init_or_resume_run(
        provider=provider,
        experiment_name=experiment_name,
        model=model,
        steps=steps,
        feedback_mode=feedback_mode,
        reasoning_effort=reasoning_effort,
        conversation_seed=conversation_seed,
        secret_format=secret_format,
        number_digits=number_digits,
        word_category=word_category,
        force_new_run=new_run,
    )
    steps_dir = run_dir / STEPS_DIR_NAME
    steps_dir.mkdir(parents=True, exist_ok=True)

    requirement = secret_requirement_text(secret_format, number_digits, word_category)
    base_prompt = (
        "Memory game:\n"
        f"At each step, output exactly {requirement} and nothing else.\n"
        "Do not output anything except that single secret token."
    )
    if secret_format == "number":
        base_prompt += (
            "\nSome numbers are explicitly forbidden and must never be output: "
            f"{', '.join(forbidden_numbers)}."
        )
    messages, history_secrets, completed, valid_steps = rebuild_context_from_steps(
        steps_dir=steps_dir,
        base_prompt=base_prompt,
        conversation_seed=conversation_seed,
        secret_format=secret_format,
        number_digits=number_digits,
        word_category=word_category,
        forbidden_numbers=forbidden_numbers,
    )
    state["completed_steps"] = completed
    state["valid_secret_steps"] = valid_steps
    state["temperature"] = temperature
    state["top_p"] = top_p
    state["max_tokens"] = max_tokens
    if "valid_word_steps" in state:
        state["valid_word_steps"] = valid_steps
    write_json(run_dir / STATE_FILE_NAME, state)

    if provider == "openrouter":
        client: OpenRouterClient | AnthropicClient = OpenRouterClient(
            OpenRouterConfig(
                model=model,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )
        )
    elif provider == "anthropic":
        client = AnthropicClient(
            AnthropicConfig(
                model=model,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}")
    append_event(run_dir, "run_started", {"run_id": state["run_id"]})

    for step_index in range(completed + 1, steps + 1):
        append_event(run_dir, "step_started", {"step_index": step_index})
        prompt = step_prompt(
            step_index,
            secret_format=secret_format,
            number_digits=number_digits,
            word_category=word_category,
            forbidden_numbers=forbidden_numbers,
        )
        messages.append({"role": "user", "content": prompt})
        conversation_for_model = [dict(m) for m in messages]

        extra_body: dict[str, Any] | None = None
        if provider == "openrouter":
            extra_body = {"reasoning": {"effort": reasoning_effort}}
        response_json = client.chat_completion_messages(messages=messages, extra_body=extra_body)
        message = response_json.get("choices", [{}])[0].get("message", {})
        content = (message.get("content", "") or "").strip()
        parsed_secret, invalid_reason = parse_secret(
            content,
            secret_format=secret_format,
            number_digits=number_digits,
            word_category=word_category,
            forbidden_numbers=forbidden_numbers,
        )
        feedback = build_feedback(
            feedback_mode,
            parsed_secret,
            history_secrets,
            invalid_reason,
            secret_format=secret_format,
            number_digits=number_digits,
            word_category=word_category,
            forbidden_numbers=forbidden_numbers,
        )

        if parsed_secret is not None:
            history_secrets.append(parsed_secret)
            valid_steps += 1
        pre_secret_text = "(nothing before secret data)"

        if feedback_mode == "no_feedback_repeat":
            validator_feedback = (
                f"{REDACTION_NOTICE}\n"
                f"{pre_secret_text}\n"
                "Now output the same secret again in exactly the same required format."
            )
        elif feedback_mode == "no_feedback_fresh":
            validator_feedback = (
                f"{REDACTION_NOTICE}\n"
                f"{pre_secret_text}\n"
                "Now output another secret in exactly the same required format."
            )
        else:
            validator_feedback = f"{REDACTION_NOTICE}\n{pre_secret_text}\n{feedback}"
        messages.append({"role": "user", "content": validator_feedback})

        row = {
            "status": "completed",
            "completed_at": now_iso(),
            "step_index": step_index,
            "prompt": prompt,
            "model_output": content,
            "conversation_for_model": conversation_for_model,
            "parsed_secret": parsed_secret,
            "parsed_word": parsed_secret if secret_format == "word" else None,
            "invalid_reason": invalid_reason,
            "pre_secret_text": pre_secret_text,
            "validator_feedback": validator_feedback,
            "feedback_mode": feedback_mode,
            "secret_format": secret_format,
            "number_digits": number_digits,
            "word_category": word_category,
            "forbidden_numbers": forbidden_numbers,
            "request_config": {
                "provider": provider,
                "model": model,
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
                "reasoning_effort": reasoning_effort,
            },
            "raw_response": response_json,
        }
        step_file = steps_dir / f"{step_index:04d}.json"
        write_json(step_file, row)
        append_event(
            run_dir,
            "step_completed",
            {"step_index": step_index, "valid_secret": parsed_secret is not None, "step_file": str(step_file)},
        )

        state["completed_steps"] = step_index
        state["valid_secret_steps"] = valid_steps
        state["valid_word_steps"] = valid_steps
        write_json(run_dir / STATE_FILE_NAME, state)
        print(f"step={step_index}/{steps} valid_secret={parsed_secret is not None} feedback={feedback}")

    state["status"] = "completed"
    write_json(run_dir / STATE_FILE_NAME, state)
    append_event(run_dir, "run_completed", {"run_id": state["run_id"]})
    print(f"Run complete: {run_dir}")
    print(f"Valid outputs: {state['valid_secret_steps']}/{state['completed_steps']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Memory experiment with feedback and redacted history.")
    parser.add_argument("--provider", default=DEFAULT_PROVIDER, choices=["anthropic", "openrouter"])
    parser.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT_NAME)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument(
        "--feedback-mode",
        choices=["previous", "all_previous", "no_feedback_repeat", "no_feedback_fresh"],
        required=True,
    )
    parser.add_argument("--new-run", action="store_true")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument(
        "--conversation-seed",
        default=DEFAULT_CONVERSATION_SEED,
        help="Optional hardcoded seed string added as the first user message in the conversation.",
    )
    parser.add_argument(
        "--reasoning-effort",
        default=DEFAULT_REASONING_EFFORT,
        choices=["xhigh", "high", "medium", "low", "minimal", "none"],
        help="OpenRouter reasoning.effort setting.",
    )
    parser.add_argument(
        "--secret-format",
        default=DEFAULT_SECRET_FORMAT,
        choices=["word", "number", "bit", "binary", "category_word"],
        help="Format of secret token the model must output each step.",
    )
    parser.add_argument(
        "--number-digits",
        type=int,
        default=DEFAULT_NUMBER_DIGITS,
        help="Required digit count when --secret-format number.",
    )
    parser.add_argument(
        "--word-category",
        default=DEFAULT_WORD_CATEGORY,
        choices=sorted(CATEGORY_WORDS.keys()),
        help="Allowed category when --secret-format category_word.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.steps < 1:
        raise ValueError("--steps must be >= 1")
    run_experiment(
        provider=args.provider,
        experiment_name=args.experiment_name,
        model=args.model,
        steps=args.steps,
        feedback_mode=args.feedback_mode,
        new_run=args.new_run,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        reasoning_effort=args.reasoning_effort,
        conversation_seed=args.conversation_seed,
        secret_format=args.secret_format,
        number_digits=args.number_digits,
        word_category=args.word_category,
    )


if __name__ == "__main__":
    main()
