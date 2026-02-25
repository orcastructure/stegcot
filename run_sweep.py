"""Run hardcoded parameter sweeps over existing experiment scripts."""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from eval_constants import EXPERIMENT_OUTPUT_DIR


SWEEP_CATALOG_PATH = Path("hardcoded/sweep_catalog.json")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def shell_safe_arg_name(name: str) -> str:
    return f"--{name.replace('_', '-')}"


def arg_to_cli_parts(name: str, value: Any) -> list[str]:
    key = shell_safe_arg_name(name)
    if isinstance(value, bool):
        return [key] if value else []
    if value is None:
        return []
    return [key, str(value)]


def expand_grid(vary: dict[str, list[Any]]) -> list[dict[str, Any]]:
    if not vary:
        return [{}]
    keys = sorted(vary.keys())
    for key in keys:
        if not isinstance(vary[key], list) or not vary[key]:
            raise ValueError(f"vary.{key} must be a non-empty list")
    combos: list[dict[str, Any]] = []
    for values in itertools.product(*(vary[k] for k in keys)):
        combos.append(dict(zip(keys, values)))
    return combos


def build_command(script: str, fixed_args: dict[str, Any], vary_args: dict[str, Any], always_new_run: bool) -> list[str]:
    cmd = [sys.executable, script]
    merged = dict(fixed_args)
    merged.update(vary_args)
    for key in sorted(merged.keys()):
        cmd.extend(arg_to_cli_parts(key, merged[key]))
    if always_new_run and "--new-run" not in cmd:
        cmd.append("--new-run")
    return cmd


def run_sweep(sweep_name: str, max_trials: int | None) -> None:
    catalog = read_json(SWEEP_CATALOG_PATH)
    if sweep_name not in catalog:
        raise ValueError(f"Sweep '{sweep_name}' not found in {SWEEP_CATALOG_PATH}")
    sweep_cfg = catalog[sweep_name]

    script = sweep_cfg.get("script")
    if not isinstance(script, str) or not script:
        raise ValueError("sweep.script must be a non-empty string")
    if not Path(script).exists():
        raise FileNotFoundError(f"Sweep script does not exist: {script}")

    fixed_args = sweep_cfg.get("fixed_args", {})
    vary = sweep_cfg.get("vary", {})
    repeats = int(sweep_cfg.get("repeats", 1))
    always_new_run = bool(sweep_cfg.get("always_new_run", True))
    if not isinstance(fixed_args, dict):
        raise ValueError("sweep.fixed_args must be an object")
    if not isinstance(vary, dict):
        raise ValueError("sweep.vary must be an object")
    if repeats < 1:
        raise ValueError("sweep.repeats must be >= 1")

    combos = expand_grid(vary)
    trials: list[dict[str, Any]] = []
    trial_index = 0
    for combo_idx, combo in enumerate(combos, start=1):
        for repeat_idx in range(1, repeats + 1):
            trial_index += 1
            if max_trials is not None and trial_index > max_trials:
                break
            cmd = build_command(script, fixed_args=fixed_args, vary_args=combo, always_new_run=always_new_run)
            trials.append(
                {
                    "trial_index": trial_index,
                    "combo_index": combo_idx,
                    "repeat_index": repeat_idx,
                    "params": combo,
                    "command": cmd,
                }
            )
        if max_trials is not None and trial_index >= max_trials:
            break

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = Path(EXPERIMENT_OUTPUT_DIR) / "sweeps" / sweep_name / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "run_id": run_id,
        "started_at": now_iso(),
        "sweep_name": sweep_name,
        "script": script,
        "fixed_args": fixed_args,
        "vary": vary,
        "repeats": repeats,
        "always_new_run": always_new_run,
        "planned_trials": len(trials),
    }
    write_json(run_dir / "manifest.json", manifest)

    successes = 0
    failures = 0
    results: list[dict[str, Any]] = []
    for trial in trials:
        print(
            f"[trial {trial['trial_index']}/{len(trials)}] "
            f"combo={trial['combo_index']} repeat={trial['repeat_index']} params={trial['params']}"
        )
        completed = subprocess.run(trial["command"], cwd=Path.cwd(), capture_output=True, text=True)
        ok = completed.returncode == 0
        if ok:
            successes += 1
        else:
            failures += 1
        row = {
            "trial_index": trial["trial_index"],
            "combo_index": trial["combo_index"],
            "repeat_index": trial["repeat_index"],
            "params": trial["params"],
            "command": trial["command"],
            "returncode": completed.returncode,
            "succeeded": ok,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "completed_at": now_iso(),
        }
        results.append(row)
        write_json(run_dir / f"trial_{trial['trial_index']:04d}.json", row)
        status = "ok" if ok else "failed"
        print(f"  -> {status} (returncode={completed.returncode})")

    summary = {
        "run_id": run_id,
        "completed_at": now_iso(),
        "sweep_name": sweep_name,
        "planned_trials": len(trials),
        "successes": successes,
        "failures": failures,
    }
    write_json(run_dir / "summary.json", summary)

    print(f"\nSweep complete: {run_dir}")
    print(f"successes={successes} failures={failures} total={len(trials)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run named hardcoded sweeps from hardcoded/sweep_catalog.json.")
    parser.add_argument("--sweep", required=True, help="Sweep name in hardcoded/sweep_catalog.json")
    parser.add_argument(
        "--max-trials",
        type=int,
        default=None,
        help="Optional cap for quick smoke tests.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_trials is not None and args.max_trials < 1:
        raise ValueError("--max-trials must be >= 1")
    run_sweep(sweep_name=args.sweep, max_trials=args.max_trials)


if __name__ == "__main__":
    main()
