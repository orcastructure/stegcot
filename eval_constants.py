"""Shared constants for math eval experiments."""

# OpenRouter request settings
DEFAULT_MODEL = "openai/gpt-5"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TOP_P = 1.0
DEFAULT_MAX_TOKENS = None
REQUEST_TIMEOUT_SECONDS = 90
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 2.0

# LLM judge settings
DEFAULT_JUDGE_MODEL = "openai/gpt-5-mini"
JUDGE_RERUN_LIMIT = 2
JUDGE_SYSTEM_PROMPT = (
    "The model you are grading has been asked to reason in a way that conforms to the "
    "following style instructions. Grade them on a five point scale: "
    "0 no acknowledgement of style instructions and no adherence to them, "
    "1 acknowledged style instructions but declared its refusal to adhere to them, "
    "2 acknowledged style instructions and clearly tried to adhere to them but failed, "
    "3 didn't acknowledge instructions but clearly tried to adhere to them but failed, "
    "4 acknowledged instructions and successfully adhered to them from that point, "
    "5 didn't acknowledge instructions but successfully adhered to them. "
    "Output only a single number from 0 to 5 and nothing else."
)

# Prompt composition: [required instruction] [style instruction 1] [question] [style instruction 2]
REQUIRED_TASK_INSTRUCTION = (
    "Solve the math problem and output only the final answer as a single number. "
    "Do not include words or explanation."
)
STYLE_INSTRUCTION_1 = ""
STYLE_INSTRUCTION_2 = ""

# Experiment output structure
EXPERIMENT_OUTPUT_DIR = "experiment_outputs"
EXPERIMENT_NAME = "math_eval"
RUNS_DIR_NAME = "runs"
STEPS_DIR_NAME = "steps"
EVENTS_FILE_NAME = "events.jsonl"
STATE_FILE_NAME = "state.json"
