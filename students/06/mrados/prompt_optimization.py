import marimo

__generated_with = "0.16.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    from sklearn.metrics import cohen_kappa_score, accuracy_score, confusion_matrix
    import anthropic
    import os
    from pathlib import Path
    return (
        Path,
        accuracy_score,
        anthropic,
        cohen_kappa_score,
        confusion_matrix,
        mo,
        pd,
    )


@app.cell
def _(mo):
    mo.md(
        """
    # 🎯 Evaluacija LLM odgovora

    ## Zadatak: Izrada i evaluacija skupa podataka

    Ova bilježnica vodi vas kroz tri koraka:
    1. **Izrada skupa podataka** - 15-20 pitanja s odgovorima
    2. **Ground truth označavanje** - Ručno označavanje kao pass/fail
    3. **Automatska evaluacija** - Usporedba s LLM evaluatorom
    """
    )
    return


@app.cell
def _(mo):
    mo.md("""---""")
    mo.md(
        """
        ## 📝 Korak 1: Izrada skupa podataka

        Izaberite temu koju dobro poznajete i upišite 15-20 pitanja s odgovorima.
        """
    )
    return


@app.cell
def _(mo):
    topic_input = mo.ui.text(
        placeholder="Unesite temu (npr. 'Programiranje u Pythonu')",
        label="Tema vašeg skupa podataka:",
        full_width=True
    )
    topic_input
    return (topic_input,)


@app.cell
def _(mo, pd):
    # Predložak za početni skup podataka
    sample_data = {
        'question': [
            'Što je lista u Pythonu?',
            'Kako se definira funkcija?',
            'Što radi append() metoda?',
        ],
        'answer': [
            'Lista je struktura podataka koja može sadržavati više elemenata.',
            'Funkcija se definira s def keyword.',
            'append() dodaje element na kraj liste.',
        ]
    }

    data_editor = mo.ui.dataframe(
        pd.DataFrame(sample_data),
        label="Unesite vaša pitanja i odgovore:",
        on_change=lambda df: df
    )

    mo.md(f"""
    ### Unesite podatke:

    {data_editor}

    💡 **Savjet**: Dodajte redove klikom na + ili uredite postojeće podatke.
    """)
    return (data_editor,)


@app.cell
def _(data_editor, mo, pd):
    # Pregled unesenih podataka
    current_data = data_editor.value if data_editor.value is not None else pd.DataFrame()

    if len(current_data) > 0:
        mo.md(f"""
        ✅ **Trenutno imate {len(current_data)} pitanja.**

        {'⚠️ Trebate barem 15 pitanja za zadatak.' if len(current_data) < 15 else '✨ Odlično! Možete prijeći na sljedeći korak.'}
        """)
    else:
        mo.md("⏳ Unesite podatke u tablicu iznad...")
    return (current_data,)


@app.cell
def _(mo):
    mo.md("""---""")
    mo.md(
        """
        ## ✓ Korak 2: Ground Truth označavanje

        Označite svaki odgovor kao **pass** ili **fail** i dodajte kratko obrazloženje.
        """
    )
    return


@app.cell
def _(current_data, mo):
    # Dodaj kolone za ground truth ako ne postoje
    if len(current_data) > 0:
        gt_data = current_data.copy()
        if 'ground_truth' not in gt_data.columns:
            gt_data['ground_truth'] = 'pass'
        if 'explanation' not in gt_data.columns:
            gt_data['explanation'] = ''

        gt_editor = mo.ui.dataframe(
            gt_data,
            label="Označite odgovore kao pass/fail:",
        )

        mo.md(f"""
        ### Ground Truth označavanje:

        {gt_editor}

        💡 **Upute**: 
        - U kolonu `ground_truth` unesite **pass** ili **fail**
        - U kolonu `explanation` unesite kratko obrazloženje
        """)
    else:
        mo.md("⏳ Prvo unesite podatke u Koraku 1...")
        gt_editor = None
    return (gt_editor,)


@app.cell
def _(gt_editor, mo, pd):
    # Provjera ground truth podataka
    gt_current = gt_editor.value if gt_editor is not None and gt_editor.value is not None else pd.DataFrame()

    if len(gt_current) > 0:
        pass_count = (gt_current['ground_truth'] == 'pass').sum()
        fail_count = (gt_current['ground_truth'] == 'fail').sum()

        mo.md(f"""
        📊 **Statistika označavanja:**
        - ✅ Pass: {pass_count}
        - ❌ Fail: {fail_count}
        - 📝 Ukupno: {len(gt_current)}
        """)
    else:
        mo.md("")
    return (gt_current,)


@app.cell
def _(Path, gt_current, mo, topic_input):
    # Gumb za spremanje CSV-a
    save_button = mo.ui.button(
        label="💾 Spremi kao CSV",
        disabled=len(gt_current) == 0
    )

    if save_button.value and len(gt_current) > 0:
        # Spremi CSV
        Path("data").mkdir(exist_ok=True)
        topic_name = topic_input.value.replace(
            " ", "_").lower() if topic_input.value else "questions"
        filename = f"data/{topic_name}_questions.csv"
        gt_current.to_csv(filename, index=False, encoding='utf-8')

        mo.md(f"""
        {save_button}

        ✅ **Podaci spremljeni u: `{filename}`**
        """)
    else:
        mo.md(f"{save_button}")
    return


@app.cell
def _(mo):
    mo.md("""---""")
    mo.md(
        """
        ## 🤖 Korak 3: Automatska evaluacija

        Koristimo Claude API za automatsku evaluaciju odgovora.
        """
    )
    return


@app.cell
def _(mo):
    api_key_input = mo.ui.text(
        placeholder="Unesite Anthropic API ključ",
        label="Anthropic API Key:",
        kind="password",
        full_width=True
    )

    mo.md(f"""
    ### Postavke API-ja:

    {api_key_input}

    💡 API ključ možete dobiti na: https://console.anthropic.com/
    """)
    return (api_key_input,)


@app.cell
def _(anthropic, api_key_input, gt_current, mo):
    # Funkcija za evaluaciju jednog odgovora
    def evaluate_answer(question, answer, api_key):
        try:
            client = anthropic.Anthropic(api_key=api_key)

            prompt = f"""Evaluiraj sljedeći odgovor na pitanje. Odgovori samo sa 'pass' ili 'fail' i kratkim obrazloženjem.

    Pitanje: {question}
    Odgovor: {answer}

    Format odgovora:
    Ocjena: [pass/fail]
    Obrazloženje: [1-2 rečenice]"""

            message = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}]
            )

            response = message.content[0].text

            # Parsiranje odgovora
            lines = response.strip().split('\n')
            evaluation = 'fail'
            reason = ''

            for line in lines:
                if 'Ocjena:' in line or 'ocjena:' in line:
                    evaluation = 'pass' if 'pass' in line.lower() else 'fail'
                elif 'Obrazloženje:' in line or 'obrazloženje:' in line:
                    reason = line.split(':', 1)[1].strip()

            return evaluation, reason
        except Exception as e:
            return 'error', str(e)

    # Gumb za pokretanje evaluacije
    eval_button = mo.ui.button(
        label="🚀 Pokreni evaluaciju",
        disabled=len(gt_current) == 0 or not api_key_input.value
    )

    mo.md(f"""
    {eval_button}
    """)
    return eval_button, evaluate_answer


@app.cell
def _(api_key_input, eval_button, evaluate_answer, gt_current, mo, pd):
    if eval_button.value and len(gt_current) > 0:
        mo.md("⏳ **Evaluacija u tijeku... Molimo pričekajte.**")

        results = []
        for idx, row in gt_current.iterrows():
            evaluation, reason = evaluate_answer(
                row['question'],
                row['answer'],
                api_key_input.value
            )
            results.append({
                'question': row['question'],
                'answer': row['answer'],
                'ground_truth': row['ground_truth'],
                'llm_evaluation': evaluation,
                'llm_reason': reason
            })

        results_df = pd.DataFrame(results)
        mo.md("✅ **Evaluacija završena!**")
    else:
        results_df = pd.DataFrame()
        mo.md("")
    return (results_df,)


@app.cell
def _(accuracy_score, cohen_kappa_score, confusion_matrix, mo, results_df):
    # Prikaz rezultata i metrika
    if len(results_df) > 0:
        # Izračun metrika
        y_true = results_df['ground_truth']
        y_pred = results_df['llm_evaluation']

        accuracy = accuracy_score(y_true, y_pred)
        kappa = cohen_kappa_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred, labels=['pass', 'fail'])

        mo.md(f"""
        ## 📊 Rezultati evaluacije

        ### Metrike:
        - **Točnost (Accuracy)**: {accuracy:.2%}
        - **Cohenova Kappa**: {kappa:.3f}

        ### Confusion Matrix:
        ```
                      Predicted
                    Pass    Fail
        Actual Pass   {cm[0][0]}      {cm[0][1]}
               Fail   {cm[1][0]}      {cm[1][1]}
        ```

        ### Interpretacija Cohen's Kappa:
        - **< 0.00**: Nema slaganja
        - **0.00 - 0.20**: Zanemarivo slaganje
        - **0.21 - 0.40**: Minimalno slaganje
        - **0.41 - 0.60**: Umjereno slaganje
        - **0.61 - 0.80**: Značajno slaganje
        - **0.81 - 1.00**: Gotovo savršeno slaganje

        **Vaš rezultat ({kappa:.3f})**: {
            'Gotovo savršeno slaganje! 🎉' if kappa > 0.80 else
            'Značajno slaganje! ✅' if kappa > 0.60 else
            'Umjereno slaganje. 🤔' if kappa > 0.40 else
            'Potrebno poboljšanje. ⚠️'
        }
        """)

        # Prikaz detaljnih rezultata
        mo.ui.table(results_df, label="Detaljni rezultati:")
    else:
        mo.md("")
    return


@app.cell
def _(mo, results_df):
    # Opcija za spremanje rezultata
    if len(results_df) > 0:
        save_results_button = mo.ui.button(label="💾 Spremi rezultate")

        if save_results_button.value:
            results_df.to_csv("data/evaluation_results.csv",
                              index=False, encoding='utf-8')
            mo.md(f"""
            {save_results_button}

            ✅ **Rezultati spremljeni u: `data/evaluation_results.csv`**
            """)
        else:
            mo.md(f"{save_results_button}")
    else:
        mo.md("")
    return


@app.cell
def _(mo):
    mo.md(
        """
    ---

    ## 🎓 Zaključak

    Završili ste sve tri koraka:
    1. ✅ Izradili ste skup podataka
    2. ✅ Označili ste "ground truth"
    3. ✅ Pokrenuli ste evaluaciju i dobili metrike

    **Što dalje?**
    - Analizirajte gdje se evaluator nije slagao s vašim oznakama
    - Razmislite zašto postoje razlike
    - Pokušajte poboljšati prompt za evaluaciju
    """
    )
    return


if __name__ == "__main__":
    app.run()
