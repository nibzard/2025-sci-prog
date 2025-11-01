import marimo

__generated_with = "0.16.5"
app = marimo.App()


@app.cell
def _():
    import os
    import requests
    from dotenv import load_dotenv
    import google.generativeai as genai
    return genai, load_dotenv, os, requests


@app.cell
def _():
    import marimo as _mo
    
    _mo.md("# 🔑 API Keys Configuration")
    return


@app.cell
def _(load_dotenv, os):
    import marimo as _mo
    
    # Try to load from .env first
    from pathlib import Path
    load_dotenv()
    try:
        load_dotenv(dotenv_path=Path(__file__).with_name(".env"), override=False)
    except Exception:
        pass
    
    # Get values from .env or empty string
    steel_default = os.getenv("STEEL_API_KEY", "")
    google_default = os.getenv("GOOGLE_API_KEY", "")
    
    # Create input fields
    steel_input = _mo.ui.text(
        label="Steel API Key",
        placeholder="ste-...",
        value=steel_default,
        kind="password"
    )
    
    google_input = _mo.ui.text(
        label="Google AI API Key",
        placeholder="AIzaSy...",
        value=google_default,
        kind="password"
    )
    
    _mo.vstack([
        _mo.md("Unesite API ključeve (ili će se automatski učitati iz .env):"),
        steel_input,
        google_input
    ])
    return steel_input, google_input, steel_default, google_default


@app.cell
def _():
    import marimo as _mo
    
    _mo.md("---\n\n# 📰 Scraper Configuration")
    return


@app.cell
def _():
    import marimo as _mo

    url_input = _mo.ui.text(
        label="URL",
        placeholder="https://news.ycombinator.com/",
        value="https://news.ycombinator.com/"
    )
    run_btn = _mo.ui.run_button(label="Scrape & Analyze HN")
    _mo.hstack([url_input, run_btn])
    return run_btn, url_input


@app.cell
def _(genai, requests, run_btn, url_input, steel_input, google_input):
    import marimo as _mo

    def steel_fetch_text(url: str, api_key: str) -> str:
        if not api_key:
            return "ERROR: STEEL_API_KEY nije postavljen."
        try:
            import json
            from bs4 import BeautifulSoup
            # Steel v1 scrape via POST with header 'steel-api-key'
            resp = requests.post(
                "https://api.steel.dev/v1/scrape",
                headers={
                    "steel-api-key": api_key,
                    "content-type": "application/json",
                    "accept": "application/json",
                },
                json={
                    "url": url,
                    "extract": "text",
                    "useProxy": False,
                    "delay": 1,
                    "fullPage": True,
                    "region": "",
                },
            )
            ct = resp.headers.get("content-type", "")
            if "application/json" in ct:
                data = resp.json()
                if isinstance(data, dict):
                    # Prefer direct text-like fields
                    for key in ("text", "content", "extracted_text"):
                        val = data.get(key)
                        if isinstance(val, str) and val.strip():
                            return val
                    # Steel often nests HTML under content.html
                    content = data.get("content")
                    if isinstance(content, dict):
                        html = content.get("html") or content.get("body")
                        if isinstance(html, str) and html.strip():
                            soup = BeautifulSoup(html, "html.parser")
                            return soup.get_text(" ", strip=True)
                    # Top-level html
                    html = data.get("html")
                    if isinstance(html, str) and html.strip():
                        soup = BeautifulSoup(html, "html.parser")
                        return soup.get_text(" ", strip=True)
                    # Fallback: stringify JSON
                    return json.dumps(data, ensure_ascii=False)
                if isinstance(data, list):
                    return "\n".join(str(item) for item in data)
                return str(data)
            return resp.text
        except Exception as e:
            return f"ERROR contacting Steel: {e}"

    def gemini_analyze_hn(text: str, api_key: str) -> str:
        if not api_key:
            return "ERROR: GOOGLE_API_KEY nije postavljen."
        try:
            genai.configure(api_key=api_key)
            # Dynamically find a model that supports generateContent
            try:
                models = list(genai.list_models())
            except Exception:
                models = []
            candidate_models = []
            for m in models:
                try:
                    methods = getattr(
                        m, "supported_generation_methods", []) or []
                    if "generateContent" in methods:
                        candidate_models.append(getattr(m, "name", ""))
                except Exception:
                    continue
            # Ensure some reasonable defaults at the end
            candidate_models += [
                "gemini-1.5-flash-latest",
                "gemini-1.5-flash",
                "gemini-1.5-pro-latest",
            ]

            prompt = (
                "Analiziraj sljedeći tekst sa Hacker News naslovne stranice i izvuci TOP 10 najvažnijih vijesti.\n\n"
                "Za svaku vijest navedi:\n"
                "- Naslov vijesti\n"
                "- Broj bodova (points) ako je vidljiv\n"
                "- Broj komentara ako je vidljiv\n"
                "- Kratki opis (1 rečenica) o čemu se radi\n\n"
                "Na kraju napiši sažetak (2-3 rečenice): koje su glavne teme danas na Hacker News?\n\n"
                "Formatiraj odgovor kao listu s bullet točkama na hrvatskom jeziku.\n"
                "Nemoj koristiti JSON ili code blokove, samo čisti tekst.\n\n"
                f"Tekst:\n{text[:8000]}"
            )

            last_err = None
            for model_name in candidate_models:
                try:
                    if not model_name:
                        continue
                    model = genai.GenerativeModel(model_name)
                    resp = model.generate_content(prompt)
                    text_result = getattr(
                        resp, "text", "") or "(prazan odgovor)"
                    return text_result
                except Exception as inner_e:
                    last_err = inner_e
                    continue
            if last_err is not None:
                return f"ERROR from Gemini: {last_err}"
            return "(prazan odgovor)"
        except Exception as e:
            return f"ERROR from Gemini: {e}"

    view = _mo.md(
        "Kliknite 'Scrape & Analyze HN' za analizu Hacker News naslovne stranice.")

    if run_btn.value:
        # Get keys from input fields
        steel_key = steel_input.value.strip()
        google_key = google_input.value.strip()
        
        url = (url_input.value or "").strip()
        if not url:
            view = _mo.md("❌ Molimo unesite URL.")
        elif not steel_key or not google_key:
            view = _mo.md(f"❌ Missing API keys!\n\nSteel key present: {bool(steel_key)}\n\nGoogle key present: {bool(google_key)}\n\n**Molimo unesite oba API ključa u polja iznad.**")
        else:
            view = _mo.md("⏳ Dohvaćam stranicu sa Steel API...")
            steel_text = steel_fetch_text(url, steel_key)

            if not isinstance(steel_text, str):
                steel_text = str(steel_text)

            if steel_text.startswith("ERROR"):
                view = _mo.md(f"❌ {steel_text}")
            else:
                # Show excerpt
                excerpt = (steel_text or "").strip()[:800]

                # Analyze with Gemini
                llm_out = gemini_analyze_hn(
                    steel_text, google_key) if steel_text else "(nema teksta)"

                results_text = (
                    f"# 📰 Hacker News Scraper Rezultati\n\n"
                    f"**Scraped URL:** `{url}`\n\n"
                    f"---\n\n"
                    f"## Steel Output (isječak, prvih 800 znakova)\n\n"
                    f"```\n{excerpt}\n```\n\n"
                    f"---\n\n"
                    f"## 🤖 LLM Analiza (Google AI)\n\n"
                    f"{llm_out}\n\n"
                    f"---\n\n"
                    f"*Duljina ukupnog teksta: {len(steel_text)} znakova*"
                )
                
                # Create download button for Markdown
                download_btn = _mo.download(
                    data=results_text.encode('utf-8'),
                    filename="hacker_news_results.md",
                    mimetype="text/markdown",
                    label="⬇️ Download Results as Markdown"
                )
                
                view = _mo.vstack([
                    _mo.md(results_text),
                    download_btn
                ])

    view
    return steel_fetch_text, gemini_analyze_hn, view


if __name__ == "__main__":
    app.run()