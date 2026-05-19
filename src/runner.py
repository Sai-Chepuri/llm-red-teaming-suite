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

from .evaluator import evaluate_response

from config.settings import (
    DEFAULT_MODEL,
    SUPPORTED_MODELS,
    DATA_DIR,
)

load_dotenv()

# Modern Dynamic SDK Initialization


def initialize_model(model_key):
    """
    Initialize model runtime dynamically based on provider configuration.
    """

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


def load_data(path):
    with open(path, "r") as f:
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


def run_and_evaluate(data_path, model_runtime, model_key):
    data = load_data(data_path)
    results = []

    safe_config = sanitize_model_config(
        SUPPORTED_MODELS[model_key]
    )

    print(f"Testing {len(data)} items with {safe_config}...")
    for item in tqdm(data):
        # output = call_model(item["input"])
        output = call_model(
            item["input"],
            model_runtime
        )

        # FIXED: Call the imported evaluator function
        eval_result = evaluate_response(
            item["input"],
            output,
            item["expected_behavior"]
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
            "reason": eval_result.get("reason", "N/A")
        })
    return results


def run_all_datasets(data_dir="data", model_runtime=None, model_key=None):
    all_results = []
    # Ensure directory exists before listing
    if not os.path.exists(data_dir):
        print(f"Error: Directory '{data_dir}' not found.")
        return []

    for file in os.listdir(data_dir):
        if file.endswith(".json"):
            path = os.path.join(data_dir, file)
            category_name = file.replace(".json", "")
            print(f"\nRunning category: {category_name}")

            # FIXED: Used run_and_evaluate inside run_all_datasets
            results = run_and_evaluate(
                path, model_runtime, model_key)
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
        "pass_rate": round(passed / total, 2)
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
            "pass_rate": round(passed / total, 2)
        }
    return metrics


def extract_failures(results):
    return [r for r in results if r["evaluation"] == "FAIL"]


def save_full_report(results, metrics, category_metrics):
    os.makedirs("results", exist_ok=True)
    with open("results/detailed_results.json", "w") as f:
        json.dump(results, f, indent=2)
    with open("results/summary.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open("results/category_metrics.json", "w") as f:
        json.dump(category_metrics, f, indent=2)


if __name__ == "__main__":
    print("\nStarting Red Team Evaluation Pipeline\n")

    args = parse_args()
    selected_model = args.model
    print(f"\nUsing model: {selected_model}")
    model_runtime = initialize_model(
        selected_model
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
            selected_model
        )

        for r in results:
            r["category"] = args.category

    # ==========================================
    # RUN ALL CATEGORIES
    # ==========================================

    else:

        results = run_all_datasets(
            DATA_DIR,
            model_runtime,
            selected_model
        )

        if results:
            metrics = compute_metrics(results)
            category_metrics = compute_category_metrics(results)
            failures = extract_failures(results)

            print("\n📊 Overall Metrics:", metrics)
            print("\n📂 Category Metrics:")
            for k, v in category_metrics.items():
                print(f"{k}: {v}")

            print(f"\n❌ Total Failures: {len(failures)}")
            save_full_report(results, metrics, category_metrics)
            print("\n✅ Reports saved in /results folder")
        else:
            print("No results generated. Check your /data folder.")
