"""Shared constants for math eval experiments."""

# OpenRouter request settings
DEFAULT_MODEL = "openai/gpt-4o-mini"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TOP_P = 1.0
DEFAULT_MAX_TOKENS = 256
REQUEST_TIMEOUT_SECONDS = 90
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 2.0

# Prompt composition: [required instruction] [optional part 1] [question] [optional part 2]
REQUIRED_TASK_INSTRUCTION = (
    "Solve the math problem and output only the final answer as a single number. "
    "Do not include words or explanation."
)
OPTIONAL_PROMPT_PART_1 = ""
OPTIONAL_PROMPT_PART_2 = ""

# Experiment output structure
EXPERIMENT_OUTPUT_DIR = "experiment_outputs"
EXPERIMENT_NAME = "math_eval"
RUNS_DIR_NAME = "runs"
STEPS_DIR_NAME = "steps"
EVENTS_FILE_NAME = "events.jsonl"
STATE_FILE_NAME = "state.json"
