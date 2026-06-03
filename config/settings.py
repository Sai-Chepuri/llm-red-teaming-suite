import os
from dotenv import load_dotenv

load_dotenv()

# =========================================================
# MODEL CONFIGURATION
# =========================================================

DEFAULT_MODEL = "gemini-2.5-flash-lite"

SUPPORTED_MODELS = {
    # Google Gemini
    "gemini-2.5-flash-lite": {
        "provider": "google",
        "model_name": "gemini-2.5-flash-lite",
        "api_key": os.getenv("GEMINI_API_KEY"),
        "temperature": 0.1,
        "max_output_tokens": 2048,
    },

    # Anthropic Claude
    "claude-haiku-4-5": {
        "provider": "anthropic",
        "model_name": "claude-haiku-4-5",
        "api_key": os.getenv("ANTHROPIC_API_KEY"),
        "temperature": 0.1,
        "max_output_tokens": 2048,
    },

    # OpenAI
    "gpt-5.4-nano": {
        "provider": "openai",
        "model_name": "gpt-5.4-nano",
        "api_key": os.getenv("OPENAI_API_KEY"),
        "temperature": 0.1,
        "max_output_tokens": 2048,
    },
}

# Judge configuration
USE_LLM_AS_JUDGE = True

JUDGE_MODEL = "gpt-5.4-nano"

ENABLE_CROSS_MODEL_JUDGING = True

if JUDGE_MODEL not in SUPPORTED_MODELS:
    raise ValueError(
        f"Unsupported judge model: {JUDGE_MODEL}"
    )

# =========================================================
# DATASET CONFIGURATION
# =========================================================

DATA_DIR = "data"

SUPPORTED_CATEGORIES = [
    "prompt_injection",
    "jailbreak",
    "hallucination",
]

# =========================================================
# RESULTS CONFIGURATION
# =========================================================

RESULTS_DIR = "results"

SAVE_DETAILED_RESULTS = True

SAVE_CATEGORY_METRICS = True

SAVE_SUMMARY = True

# =========================================================
# LOGGING CONFIGURATION
# =========================================================

LOG_DIR = "logs"

LOG_FILE = "logs/app.log"

LOG_LEVEL = "INFO"

# =========================================================
# RETRY CONFIGURATION
# =========================================================

MAX_RETRIES = 3

RETRY_DELAY_SECONDS = 2

# =========================================================
# EVALUATION CONFIGURATION
# =========================================================

EVALUATION_MODEL = "gemini-2.5-flash-lite"

USE_LLM_AS_JUDGE = True

STRICT_JSON_MODE = False

# =========================================================
# BENCHMARK CONFIGURATION
# =========================================================

ENABLE_CATEGORY_METRICS = True

ENABLE_FAILURE_ANALYSIS = True

ENABLE_MODEL_COMPARISON = True
