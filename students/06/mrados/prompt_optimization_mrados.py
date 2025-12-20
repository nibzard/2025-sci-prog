import marimo

__generated_with = "0.16.5"
app = marimo.App(width="medium")


@app.cell
def _():
    # ============================================================
    # REAL PROMPT OPTIMIZER — PR CLASSIFICATION (GEMINI POWERED)
    # ============================================================

    from google import genai
    from collections import defaultdict
    import os

    # -------------------------
    # 0. CLIENT SETUP
    # -------------------------

    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY")
    )
    return client, defaultdict


@app.cell
def _():
    # -------------------------
    # 1. TASK DEFINITION
    # -------------------------

    LABELS = ["bug_fix", "feature", "refactor", "docs", "test"]

    TASK_DESCRIPTION = """
    Classify GitHub pull requests into exactly one category:
    - bug_fix
    - feature
    - refactor
    - docs
    - test
    """

    EVALUATION_METRIC = "macro F1 score"

    return (LABELS,)


@app.cell
def _():
    # -------------------------
    # 2. DATASET
    # -------------------------

    TEST_CASES = [
        {"text": "Fix null pointer exception when user profile is missing", "label": "bug_fix"},
        {"text": "Add OAuth login support", "label": "feature"},
        {"text": "Refactor authentication service to improve readability", "label": "refactor"},
        {"text": "Update README with installation instructions", "label": "docs"},
        {"text": "Add unit tests for payment processing", "label": "test"},
        {"text": "Fix race condition in async request handler", "label": "bug_fix"},
        {"text": "Improve folder structure without changing behavior", "label": "refactor"},
        {"text": "Add API documentation comments", "label": "docs"},
    ]
    return (TEST_CASES,)


@app.cell
def _():
    # -------------------------
    # 3. INITIAL SYSTEM PROMPT
    # -------------------------

    system_prompt = """
    You are a code review assistant.

    Task:
    Classify a GitHub pull request into exactly ONE of the following categories:
    - bug_fix
    - feature
    - refactor
    - docs
    - test

    Definitions:
    - bug_fix: fixes incorrect or broken behavior
    - feature: adds new user-facing functionality
    - refactor: restructures code without changing behavior
    - docs: changes only documentation
    - test: adds or modifies tests

    Rules:
    - Output ONLY the label
    - No explanations
    """
    return (system_prompt,)


@app.cell
def _(client):
    # -------------------------
    # 4. FLASH MODEL (CLASSIFIER)
    # -------------------------

    def flash_lite_classify(prompt: str, pr_text: str) -> str:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=f"{prompt}\n\nPull Request:\n{pr_text}",
        )

        return response.text.strip().lower()
    return (flash_lite_classify,)


@app.cell
def _(client):
    # -------------------------
    # 5. GEMINI PRO (PROMPT OPTIMIZER)
    # -------------------------

    def gemini_rewrite_prompt(prompt: str, failures: list) -> str:
        optimization_prompt = f"""
    You are an expert prompt engineer.

    CURRENT SYSTEM PROMPT:
    {prompt}

    FAILED CLASSIFICATIONS:
    {failures}

    Task:
    Rewrite the system prompt to reduce these failures.

    Rules:
    - Improve general rules and definitions
    - Do NOT reference specific examples
    - Keep output concise
    - Output ONLY the new system prompt
    """

        response = client.models.generate_content(
            model="gemini-2.5-pro-latest",
            contents=optimization_prompt,
        )

        return response.text.strip()
    return (gemini_rewrite_prompt,)


@app.cell
def _(LABELS, defaultdict):
    # -------------------------
    # 6. EVALUATION (MACRO F1)
    # -------------------------

    def compute_macro_f1(y_true, y_pred):
        tp = defaultdict(int)
        fp = defaultdict(int)
        fn = defaultdict(int)

        for t, p in zip(y_true, y_pred):
            if t == p:
                tp[t] += 1
            else:
                fp[p] += 1
                fn[t] += 1

        f1_scores = []
        for label in LABELS:
            precision = tp[label] / (tp[label] + fp[label]) if (tp[label] + fp[label]) else 0
            recall = tp[label] / (tp[label] + fn[label]) if (tp[label] + fn[label]) else 0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0
            f1_scores.append(f1)

        return sum(f1_scores) / len(f1_scores)
    return (compute_macro_f1,)


@app.cell
def _(
    TEST_CASES,
    compute_macro_f1,
    flash_lite_classify,
    gemini_rewrite_prompt,
    system_prompt,
):
    # -------------------------
    # 7. OPTIMIZATION LOOP
    # -------------------------

    MAX_ITERATIONS = 6
    TARGET_F1 = 0.90

    current_prompt = system_prompt

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n========== ITERATION {iteration} ==========")

        predictions = []
        ground_truth = []

        for case in TEST_CASES:
            pred = flash_lite_classify(current_prompt, case["text"])
            predictions.append(pred)
            ground_truth.append(case["label"])

        f1 = compute_macro_f1(ground_truth, predictions)

        print("Predictions:", predictions)
        print("F1 score:", round(f1, 3))

        if f1 >= TARGET_F1:
            print("✅ Converged")
            break

        failures = [
            {
                "text": case["text"],
                "true": t,
                "pred": p,
            }
            for case, t, p in zip(TEST_CASES, ground_truth, predictions)
            if t != p
        ]

        print("Failures:", len(failures))

        current_prompt = gemini_rewrite_prompt(current_prompt, failures)


    print("\n========== FINAL OPTIMIZED PROMPT ==========")
    print(current_prompt)
    return


if __name__ == "__main__":
    app.run()
