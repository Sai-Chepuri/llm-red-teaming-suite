import json
import os
from dotenv import load_dotenv
from google import genai
from tqdm import tqdm
from evaluator import evaluate_response
from collections import defaultdict

load_dotenv()

# Modern SDK Initialization
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Use the latest model ID
MODEL_ID = "gemini-2.5-flash-lite"

def load_data(path):
    with open(path, "r") as f:
        return json.load(f)

def call_model(prompt):
    try:
        # FIXED: Use client.models interface
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=prompt
        )
        return response.text if response.text else "ERROR: No text response"
    except Exception as e:
        return f"ERROR: {str(e)}"

def run_and_evaluate(data_path):
    data = load_data(data_path)
    results = []

    print(f"Testing {len(data)} items with {MODEL_ID}...")
    for item in tqdm(data):
        output = call_model(item["input"])
        
        # FIXED: Call the imported evaluator function
        eval_result = evaluate_response(item["input"], output, item["expected_behavior"])

        results.append({
            "id": item["id"],
            "input": item["input"],
            "output": output,
            "expected": item["expected_behavior"],
            "evaluation": eval_result.get("result", "ERROR"),
            "reason": eval_result.get("reason", "N/A")
        })
    return results

def run_all_datasets(data_dir="data"):
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
            results = run_and_evaluate(path)
            for r in results:
                r["category"] = category_name
            all_results.extend(results)
    return all_results

def compute_metrics(results):
    total = len(results)
    if total == 0: return {"total": 0, "pass_rate": 0}
    
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
    print("\n🚀 Starting Red Team Evaluation Pipeline\n")
    
    results = run_all_datasets("data")
    
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
