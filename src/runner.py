import json
import os
import argparse

from dotenv import load_dotenv
from tqdm import tqdm
from collections import defaultdict

from anthropic import Anthropic
from openai import OpenAI
from google import genai
from google.genai import types

from config import settings

from .evaluator import evaluate_response

from config.settings import (
    DEFAULT_MODEL,
    SUPPORTED_MODELS,
    DATA_DIR,
)

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

    if override_model:

        if override_model == target_model_key:
            raise ValueError(
                "Judge model cannot be "
                "the same as target model."
            )

        return override_model

    # ==========================================
    # Config default
    # ==========================================

    default_judge = getattr(
        settings,
        "DEFAULT_JUDGE_MODEL",
        None
    )

    if default_judge:

        if default_judge == target_model_key:
            raise ValueError(
                "DEFAULT_JUDGE_MODEL cannot "
                "match target model."
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
        default=DEFAULT_MODEL,
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

def run_and_evaluate(data_path, model_runtime, model_key, judge_runtime=None):

    data = load_data(data_path)

    if not data:
        print(f"Warning: Empty dataset -> {data_path}")
        return []

    results = []

    safe_config = sanitize_model_config(
        # SUPPORTED_MODELS[model_key]
        model_runtime["config"]
    )

    print(f"Testing {len(data)} items with {safe_config}...")
    for item in tqdm(data):

        # ==========================================
        # DATASET SCHEMA VALIDATION
        # ==========================================

        required_fields = {
            "id",
            "input",
            "expected_behavior"
        }

        missing = required_fields - item.keys()

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
            item["input"],
            model_runtime
        )

        # ==========================================
        # EVALUATION
        # ==========================================

        eval_result = evaluate_response(
            item["input"],
            output,
            item["expected_behavior"],
            judge_runtime=judge_runtime
        )
        if not eval_result:
            eval_result = {
                "result": "ERROR",
                "reason": "Evaluation returned None"
            }

        results.append({
            "id": item["id"],
            "input": item["input"],
            "output": output,
            "expected": item["expected_behavior"],
            "evaluation": eval_result.get("result", "ERROR"),
            "reason": eval_result.get("reason", "N/A"),
            "target_model": model_key,
            "judge_model": (
                judge_runtime["config"]["model_name"]
                if judge_runtime else None
            )
        })
    return results


# def run_all_datasets(data_dir, model_runtime=None, model_key=None):
def run_all_datasets(data_dir, model_runtime=None, model_key=None, judge_runtime=None, judge_model_key=None):

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

            # FIXED: Used run_and_evaluate inside run_all_datasets
            results = run_and_evaluate(
                path,
                model_runtime=model_runtime,
                model_key=model_key,
                judge_runtime=judge_runtime
            )
            for r in results:
                r["category"] = category_name
            all_results.extend(results)
    return all_results


def compute_metrics(results):
    total = len(results)
    if total == 0:
        return {"total": 0, "pass_rate": 0}

    passed = sum(1 for r in results if r["evaluation"] == "PASS")
    return {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": round((passed / total) * 100, 2)
    }


def compute_category_metrics(results):
    category_stats = defaultdict(list)
    for r in results:
        category_stats[r["category"]].append(r)

    metrics = {}
    for category, items in category_stats.items():
        total = len(items)
        passed = sum(1 for i in items if i["evaluation"] == "PASS")
        metrics[category] = {
            "total": total,
            "passed": passed,
            "failed": total - passed,
            "pass_rate": round((passed / total) * 100, 2)
        }
    return metrics


def extract_failures(results):
    return [r for r in results if r["evaluation"] == "FAIL"]


def save_full_report(results, metrics, category_metrics):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "detailed_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(RESULTS_DIR, "category_metrics.json"), "w") as f:
        json.dump(category_metrics, f, indent=2)


def benchmark_all_models():
    # ==========================================
    # Run evaluation pipeline across all configured models.
    # ==========================================

    benchmark_results = {}

    for model_key in SUPPORTED_MODELS.keys():

        print(f"\n Benchmarking: {model_key}")

        model_runtime = initialize_model(model_key)
        judge_runtime = None
        judge_model_key = None

        if settings.USE_LLM_AS_JUDGE:

            judge_model_key = select_judge_model(
                target_model_key=model_key
            )

            print(
                f"Judge model: "
                f"{judge_model_key}"
            )

            judge_runtime = initialize_model(
                judge_model_key
            )

        # Run evaluation
        # results = run_all_datasets(
        #     DATA_DIR,
        #     model_runtime,
        #     model_key
        # )

        results = run_all_datasets(
            DATA_DIR,
            model_runtime,
            model_key,
            judge_runtime=judge_runtime,
            judge_model_key=judge_model_key
        )
        # Compute metrics
        overall_metrics = compute_metrics(results)

        category_metrics = compute_category_metrics(results)

        benchmark_results[model_key] = {
            "overall": overall_metrics,
            "categories": category_metrics
        }

    return benchmark_results


def print_benchmark_table(benchmark_results):

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

    for model_name, data in benchmark_results.items():

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


def save_benchmark_results(benchmark_results):

    os.makedirs(RESULTS_DIR, exist_ok=True)

    with open(
        os.path.join(RESULTS_DIR, "model_benchmarks.json"),
        "w"
    ) as f:

        json.dump(
            benchmark_results,
            f,
            indent=2
        )


if __name__ == "__main__":
    print("\nStarting Red Team Evaluation Pipeline\n")

    args = parse_args()
    # selected_model = args.model
    # print(f"\nUsing model: {selected_model}")
    # model_runtime = initialize_model(
    #     selected_model
    # )

    # =====================================================
    # COMPARE ALL MODELS
    # =====================================================

    if args.compare_all:

        benchmark_results = benchmark_all_models()

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

        selected_model = args.model

        print(f"\nAnswer Model: {selected_model}")

        model_runtime = initialize_model(
            selected_model
        )

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
        # ==========================================

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

            results = run_and_evaluate(
                dataset_path,
                model_runtime,
                selected_model,
                judge_runtime=judge_runtime
            )

            for r in results:
                r["category"] = args.category

        # ==========================================
        # RUN ALL CATEGORIES
        # ==========================================

        else:

            # results = run_all_datasets(
            #     DATA_DIR,
            #     model_runtime,
            #     selected_model
            # )

            results = run_all_datasets(
                DATA_DIR,
                model_runtime,
                selected_model,
                judge_runtime=judge_runtime,
                judge_model_key=judge_model_key
            )

        # ==========================================
        # METRICS + REPORTS
        # ==========================================

        if results:
            metrics = compute_metrics(results)
            category_metrics = compute_category_metrics(results)
            failures = extract_failures(results)

            print("\n Overall Metrics:", metrics)
            print("\n Category Metrics:")
            for k, v in category_metrics.items():
                print(f"{k}: {v}")

            print(f"\n❌ Total Failures: {len(failures)}")

            save_full_report(results, metrics, category_metrics)
            print("\n✅ Reports saved in /results folder")
        else:
            print("No results generated. Check your /data folder.")
