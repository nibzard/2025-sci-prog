import marimo

__generated_with = "0.16.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import os
    from dotenv import load_dotenv
    load_dotenv()

    import google.generativeai as genai
    import json

    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

    def bleu_score(reference, candidate):
        ref_tokens = reference.lower().split()
        cand_tokens = candidate.lower().split()
        if len(cand_tokens) == 0:
            return 0.0
        matches = sum(1 for w in cand_tokens if w in ref_tokens)
        precision = matches / len(cand_tokens)
        brevity = min(1.0, len(cand_tokens) / len(ref_tokens))
        return precision * brevity

    def run_student_model(prompt):
        model = genai.GenerativeModel("gemini-2.5-flash-lite")
        r = model.generate_content(prompt)
        return r.text

    def run_judge_model(query):
        model = genai.GenerativeModel("gemini-2.5-pro")
        r = model.generate_content(query)
        return r.text

    def build_judge_query(original_prompt, student_answer):
        query = f"""
            You are a prompt optimizer. Your job: refine the prompt for a cheaper translation model.

            TASK: Croatian-English Translation Translate technical docs
            Eval: BLEU + terminology ("funkcija" → "function")

            Return JSON ONLY with keys:
            - optimized_prompt
            - feedback
            - rationale

            Original prompt: "{original_prompt}"
            Student output: "{student_answer}"
            """
        return query

    def optimize_prompt_iteration(original_prompt, student_outputs):
        judge_query = build_judge_query(original_prompt, student_outputs)
        judge_raw = run_judge_model(judge_query)
        try:
            return json.loads(judge_raw)
        except:
            return {"optimized_prompt": original_prompt, "feedback": "parse error", "rationale": "invalid JSON"}

    initial_prompt = "Prevedi na hrvatski pazeći na stručne izraze:"

    tests = [
        "Data science is an interdisciplinary academic field that uses statistics.",
        "A function maps inputs to outputs in a predictable mathematical way.",
        "The processor executes instructions defined by the algorithm."
    ]

    current_prompt = initial_prompt
    history = []

    max_iters = 5
    for _ in range(max_iters):
        scores = []
        outputs = []
        for text in tests:
            student_input = current_prompt + "\nPrevedi ovo na hrvatski: " + text
            out = run_student_model(student_input)
            outputs.append(out)
            scores.append(bleu_score(text.lower(), out.lower()))
        avg_score = sum(scores) / len(scores)
        history.append((current_prompt, avg_score))
        student_joined = " ||| ".join(outputs)
        updated = optimize_prompt_iteration(current_prompt, student_joined)
        if "optimized_prompt" in updated:
            if updated["optimized_prompt"].strip() == current_prompt.strip():
                break
            current_prompt = updated["optimized_prompt"]
        else:
            break

    print("=== KONACNI OPTIMIZIRANI PROMPT ===")
    print(current_prompt)
    print("\n=== ITERACIJSKA POVIJEST (prompt + BLEU) ===")
    for p, s in history:
        print("\nPROMPT:\n", p)
        print("BLEU:", s)

    return


if __name__ == "__main__":
    app.run()
