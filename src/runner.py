from config import settings
from config.settings import (
    DEFAULT_MODEL,
    SUPPORTED_MODELS,
    DATA_DIR,
)
import json
import os
import sys
import argparse
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm
from collections import defaultdict

from anthropic import Anthropic
from openai import OpenAI
from google import genai
from google.genai import types

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


try:
    from .evaluator import evaluate_response
except ImportError:
    from evaluator import evaluate_response


RESULTS_DIR = "results"
load_dotenv()

# Modern Dynamic SDK Initialization


def initialize_model(model_key):
    """
    Initialize model runtime dynamically based on provider configuration.
    """

    if model_key not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unsupported model '{model_key}'. "
            f"Supported models: {list(SUPPORTED_MODELS.keys())}"
        )

    model_config = SUPPORTED_MODELS[model_key]
    provider = model_config["provider"]

    # =====================================================
    # GOOGLE GEMINI
    # =====================================================

    if provider == "google":

        client = genai.Client(
            api_key=model_config["api_key"]
        )

    # =====================================================
    # OPENAI
    # =====================================================

    elif provider == "openai":

        client = OpenAI(
            api_key=model_config["api_key"]
        )

    # =====================================================
    # ANTHROPIC
    # =====================================================

    elif provider == "anthropic":

        client = Anthropic(
            api_key=model_config["api_key"]
        )

    else:
        raise ValueError(f"Unsupported provider: {provider}")

    return {
        "provider": provider,
        "client": client,
        "config": model_config
    }


def select_judge_model(target_model_key, override_model=None):
    """
    Select a judge model that is NOT
    the same as the target model.
    """

    # ==========================================
    # CLI override has highest priority
    # ==========================================

    # Normalize common aliases and validate override
    ALIASES = {
        "claude-3-haiku": "claude-haiku-4-5",
        "claude-haiku": "claude-haiku-4-5",
        "gpt-5.4": "gpt-5.4-nano",
        "gemini-2.5": "gemini-2.5-flash-lite",
    }

    if override_model:
        # Direct alias or exact match
        normalized = ALIASES.get(override_model, override_model)

        if normalized == target_model_key:
            raise ValueError(
                f"Judge model cannot be same as target model: {target_model_key}"
            )

        if normalized in SUPPORTED_MODELS:
            print(f"Using judge override: {override_model} -> {normalized}")
            return normalized

        # Fuzzy match: sanitize strings and try to find a single supported key
        def _sanitize(name: str) -> str:
            return ''.join(ch for ch in (name or '').lower() if ch.isalnum())

        target_sanitized = _sanitize(override_model)
        matches = []
        for key in SUPPORTED_MODELS.keys():
            if target_sanitized in _sanitize(key) or _sanitize(key) in target_sanitized:
                matches.append(key)

        if len(matches) == 1:
            chosen = matches[0]
            if chosen == target_model_key:
                raise ValueError(
                    f"Judge model cannot be same as target model: {target_model_key}"
                )
            print(
                f"Fuzzy matched judge override: {override_model} -> {chosen}")
            return chosen

        raise ValueError(
            f"Unknown judge model override '{override_model}'. "
            f"Supported models: {list(SUPPORTED_MODELS.keys())}. "
            f"Candidates matched: {matches}"
        )

    # ==========================================
    # Config default
    # ==========================================

    # Support both DEFAULT_JUDGE_MODEL and legacy JUDGE_MODEL in settings
    default_judge = getattr(settings, "DEFAULT_JUDGE_MODEL", None) or getattr(
        settings, "JUDGE_MODEL", None)

    if default_judge:
        if default_judge == target_model_key:
            raise ValueError(
                f"DEFAULT_JUDGE_MODEL cannot "
                f"match target model: {target_model_key}"
            )

        return default_judge

    # ==========================================
    # Automatic fallback selection
    # ==========================================

    preferred_order = [
        "gpt-5.4-nano",
        "claude-haiku-4-5",
        "gemini-2.5-flash-lite"
    ]

    for candidate in preferred_order:

        if (
            candidate != target_model_key
            and candidate in SUPPORTED_MODELS
        ):
            return candidate

    raise ValueError(
        "No valid judge model available."
    )


def load_data(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def call_model(prompt, model_runtime):
    """
    Unified model calling interface
    for all providers.
    """

    provider = model_runtime["provider"]
    client = model_runtime["client"]
    config = model_runtime["config"]

    try:

        # =================================================
        # GOOGLE GEMINI
        # =================================================

        if provider == "google":

            response = client.models.generate_content(
                model=config["model_name"],
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=config["temperature"],
                    max_output_tokens=config["max_output_tokens"],
                    thinking_config={
                        "include_thoughts": False
                    }
                )
            )

            return response.text if response.text else "ERROR: Empty response"

        # =================================================
        # OPENAI
        # =================================================

        elif provider == "openai":

            response = client.chat.completions.create(
                model=config["model_name"],
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_completion_tokens=config["max_output_tokens"],
                temperature=config["temperature"],
            )

            return response.choices[0].message.content

        # =================================================
        # ANTHROPIC
        # =================================================

        elif provider == "anthropic":

            response = client.messages.create(
                model=config["model_name"],
                max_tokens=config["max_output_tokens"],
                temperature=config["temperature"],
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

            if not response.content or not response.content[0].text:
                return "ERROR: Empty response"

            return response.content[0].text

        else:
            return "ERROR: Unsupported provider"

    except Exception as e:
        return f"ERROR: {str(e)}"


# def call_model(prompt):
#     try:
#         # FIXED: Use client.models interface
#         response = client.models.generate_content(
#             model=MODEL_ID,
#             contents=prompt
#         )
#         return response.text if response.text else "ERROR: No text response"
#     except Exception as e:
#         return f"ERROR: {str(e)}"

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--category",
        type=str,
        help="Run specific attack category"
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model to evaluate"
    )

    parser.add_argument(
        "--compare-all",
        action="store_true",
        help="Run benchmark across all configured models"
    )

    parser.add_argument(
        "--judge-model",
        type=str,
        default=None,
        help="Optional override judge model"
    )

    return parser.parse_args()


def sanitize_model_config(model_config):
    """
    Remove sensitive fields before logging.
    """

    sensitive_keys = {
        "api_key",
        "access_token",
        "secret",
        "password"
    }

    return {
        k: v
        for k, v in model_config.items()
        if k not in sensitive_keys
    }


# def run_and_evaluate(data_path, model_runtime, model_key):

def run_and_evaluate(data_path, runtime_env, model_key, judge_runtime_env=None):

    data = load_data(data_path)

    if not data:
        print(f"Warning: Empty dataset -> {data_path}")
        return []

    results = []

    safe_config = sanitize_model_config(
        # SUPPORTED_MODELS[model_key]
        runtime_env["config"]
    )
    print(f"Testing {len(data)} items with {safe_config}...")
    for data_item in tqdm(data):

        # ==========================================
        # DATASET SCHEMA VALIDATION
        # ==========================================

        required_fields = {
            "id",
            "input",
            "expected_behavior"
        }

        missing = required_fields - data_item.keys()

        if missing:
            print(
                f"Skipping malformed item "
                f"in {data_path}: missing {missing}"
            )
            continue

        # ==========================================
        # MODEL EXECUTION
        # ==========================================

        output = call_model(
            data_item["input"],
            runtime_env
        )

        # ==========================================
        # EVALUATION
        # ==========================================

        eval_result = evaluate_response(
            data_item["input"],
            output,
            data_item["expected_behavior"],
            judge_runtime=judge_runtime_env
        )
        if not eval_result:
            eval_result = {
                "result": "ERROR",
                "reason": "Evaluation returned None"
            }

        results.append({
            "id": data_item["id"],
            "input": data_item["input"],
            "output": output,
            "expected": data_item["expected_behavior"],
            "evaluation": eval_result.get("result", "ERROR"),
            "reason": eval_result.get("reason", "N/A"),
            "target_model": model_key,
            "judge_model": (
                judge_runtime_env["config"]["model_name"]
                if judge_runtime_env else None
            )
        })
    return results


# def run_all_datasets(data_dir, model_runtime=None, model_key=None):
def run_all_datasets(data_dir, runtime_env=None, model_key=None, judge_runtime_env=None):

    all_results = []
    # Ensure directory exists before listing
    if not os.path.exists(data_dir):
        print(f"Error: Directory '{data_dir}' not found.")
        return []

    for file in sorted(os.listdir(data_dir)):
        if file.endswith(".json"):
            path = os.path.join(data_dir, file)
            category_name = file.replace(".json", "")
            print(f"\nRunning category: {category_name}")

            results = run_and_evaluate(
                path,
                runtime_env,
                model_key,
                judge_runtime_env=judge_runtime_env
            )
            for r in results:
                r["category"] = category_name
            all_results.extend(results)
    return all_results


def compute_metrics(result_list):
    total = len(result_list)
    if total == 0:
        return {"total": 0, "pass_rate": 0}

    passed = sum(1 for r in result_list if r["evaluation"] == "PASS")
    return {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": round((passed / total) * 100, 2)
    }


def compute_category_metrics(result_list):
    category_stats = defaultdict(list)
    for entry in result_list:
        category_stats[entry["category"]].append(entry)

    category_metrics = {}
    for category, items in category_stats.items():
        total = len(items)
        passed = sum(1 for i in items if i["evaluation"] == "PASS")
        category_metrics[category] = {
            "total": total,
            "passed": passed,
            "failed": total - passed,
            "pass_rate": round((passed / total) * 100, 2)
        }
    return category_metrics


def extract_failures(result_list):
    return [item for item in result_list if item["evaluation"] == "FAIL"]


def save_full_report(result_list, metrics_data, category_metrics_payload):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "detailed_results.json"), "w", encoding="utf-8") as f:
        json.dump(result_list, f, indent=2)
    with open(os.path.join(RESULTS_DIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2)
    with open(os.path.join(RESULTS_DIR, "category_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(category_metrics_payload, f, indent=2)


def benchmark_all_models(judge_override=None):
    # ==========================================
    # Run evaluation pipeline across all configured models.
    # ==========================================

    benchmark_data = {}

    for model_key in SUPPORTED_MODELS.keys():

        runtime = initialize_model(model_key)

        judge_runtime_env = None
        judge_model_key = None

        if settings.USE_LLM_AS_JUDGE:

            judge_model_key = select_judge_model(
                target_model_key=model_key,
                override_model=judge_override
            )

            print(
                f"Judge model: "
                f"{judge_model_key}"
            )

            if judge_model_key == model_key:
                raise ValueError(
                    f"Judge model cannot equal target model: {model_key}"
                )

            try:
                judge_runtime_env = initialize_model(judge_model_key)
            except Exception as e:
                print(f"Error initializing judge model {judge_model_key}: {e}")
                judge_runtime_env = None

        print(
            f"\nBenchmarking "
            f"target={model_key} "
            f"judge={judge_model_key}"
        )

        model_results = run_all_datasets(
            DATA_DIR,
            runtime,
            model_key,
            judge_runtime_env=judge_runtime_env
        )
        overall_metrics = compute_metrics(model_results)

        category_metrics = compute_category_metrics(model_results)

        benchmark_data[model_key] = {
            "overall": overall_metrics,
            "categories": category_metrics
        }

    return benchmark_data


def print_benchmark_table(benchmark_data):

    print("\n" + "=" * 90)
    print("LLM SAFETY BENCHMARK RESULTS")
    print("=" * 90)

    header = (
        f"{'MODEL':30}"
        f"{'OVERALL':15}"
        f"{'PROMPT_INJ':15}"
        f"{'JAILBREAK':15}"
        f"{'HALLUCINATION':15}"
    )

    print(header)
    print("-" * 90)

    for model_name, data in benchmark_data.items():

        overall = (
            data["overall"]["pass_rate"]
        )

        prompt_inj = (
            data["categories"]
            .get("prompt_injection", {})
            .get("pass_rate", "N/A")
        )

        jailbreak = (
            data["categories"]
            .get("jailbreak", {})
            .get("pass_rate", "N/A")
        )

        hallucination = (
            data["categories"]
            .get("hallucination", {})
            .get("pass_rate", "N/A")
        )

        row = (
            f"{model_name:30}"
            f"{str(overall):15}"
            f"{str(prompt_inj):15}"
            f"{str(jailbreak):15}"
            f"{str(hallucination):15}"
        )

        print(row)

    print("=" * 90)


def save_benchmark_results(benchmark_data):

    os.makedirs(RESULTS_DIR, exist_ok=True)

    with open(
        os.path.join(RESULTS_DIR, "model_benchmarks.json"),
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            benchmark_data,
            f,
            indent=2
        )


if __name__ == "__main__":
    print("\nStarting Red Team Evaluation Pipeline\n")

    args = parse_args()
    # Print parsed args for clarity when debugging CLI overrides
    print(
        f"Parsed args: model={args.model}, compare_all={args.compare_all}, judge_model={args.judge_model}, category={args.category}")
    # selected_model = args.model
    # print(f"\nUsing model: {selected_model}")
    # model_runtime = initialize_model(
    #     selected_model
    # )

    # =====================================================
    # COMPARE ALL MODELS
    # =====================================================

    if args.compare_all:

        benchmark_results = benchmark_all_models(
            judge_override=args.judge_model)

        print_benchmark_table(
            benchmark_results
        )

        save_benchmark_results(
            benchmark_results
        )

        print(
            "\n✅ Benchmark results saved"
        )

    # =====================================================
    # SINGLE MODEL EXECUTION
    # =====================================================

    else:

        # Determine selected model: prefer CLI override, else default
        selected_model = args.model if args.model else DEFAULT_MODEL

        print(f"\nAnswer Model: {selected_model}")

        target_runtime = initialize_model(selected_model)

        judge_runtime = None
        judge_model_key = None

        if settings.USE_LLM_AS_JUDGE:

            judge_model_key = select_judge_model(
                target_model_key=selected_model,
                override_model=args.judge_model
            )

            print(
                f"Judge model: "
                f"{judge_model_key}"
            )

            judge_runtime = initialize_model(
                judge_model_key
            )

        # ==========================================
        # RUN SINGLE CATEGORY
        # ==================================

        if args.category:

            dataset_path = os.path.join(
                DATA_DIR,
                f"{args.category}.json"
            )

            if not os.path.exists(dataset_path):
                raise FileNotFoundError(
                    f"Category dataset not found: {dataset_path}"
                )

            print(f"\nRunning category: {args.category}")

            run_results = run_and_evaluate(
                dataset_path,
                target_runtime,
                selected_model,
                judge_runtime_env=judge_runtime
            )

            for item in run_results:
                item["category"] = args.category

        else:

            run_results = run_all_datasets(
                DATA_DIR,
                target_runtime,
                selected_model,
                judge_runtime_env=judge_runtime
            )

        # ==========================================
        # METRICS + REPORTS
        # ==================================

        if run_results:
            summary_metrics = compute_metrics(run_results)
            category_metrics_data = compute_category_metrics(run_results)
            failures = extract_failures(run_results)

            print("\n Overall Metrics:", summary_metrics)
            print("\n Category Metrics:")
            for k, v in category_metrics_data.items():
                print(f"{k}: {v}")

            print(f"\n❌ Total Failures: {len(failures)}")

            save_full_report(run_results, summary_metrics,
                             category_metrics_data)
            print("\n✅ Reports saved in /results folder")
        else:
            print("No results generated. Check your /data folder.")
