from .judge_router import build_judge_prompt, call_judge_model


def evaluate_response(input_text, output, expected, judge_runtime=None):
    if judge_runtime is None:
        return {
            "result": "ERROR",
            "reason": "Judge runtime not provided"
        }

    evaluation_prompt = build_judge_prompt(
        input_text,
        output,
        expected
    )

    verdict = call_judge_model(
        evaluation_prompt,
        judge_runtime=judge_runtime)

    if isinstance(evaluation_prompt, dict):
        return evaluation_prompt

    return verdict
