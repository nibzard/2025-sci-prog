import marimo as mo


app = mo.App()


@app.cell
def _():
    import os
    import requests
    from dotenv import load_dotenv
    import google.generativeai as genai

    return os, requests, load_dotenv, genai


@app.cell
def _():
    import marimo as _mo

    url_input = _mo.ui.text(label="URL", placeholder="https://example.com", value="")
    run_btn = _mo.ui.run_button(label="Scrape & summarize")
    _mo.hstack([url_input, run_btn])
    return url_input, run_btn


@app.cell
def _(url_input, run_btn, os, requests, load_dotenv, genai):
    import marimo as _mo
    from pathlib import Path
    load_dotenv()
    # Also load .env from the script's directory to avoid CWD issues
    try:
        load_dotenv(dotenv_path=Path(__file__).with_name(".env"), override=False)
    except Exception:
        pass
    steel_key = os.getenv("STEEL_API_KEY")
    google_key = os.getenv("GOOGLE_API_KEY")

    def steel_fetch_text(url: str, api_key: str) -> str:
        if not api_key:
            return "ERROR: STEEL_API_KEY nije postavljen (.env)."
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
                    # optional flags similar to screenshot example; harmless for scrape
                    "useProxy": False,
                    "delay": 1,
                    "fullPage": True,
                    "region": "",
                },
                # no timeout
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

    def gemini_summarize(excerpt: str, api_key: str) -> str:
        if not api_key:
            return "ERROR: GOOGLE_API_KEY nije postavljen (.env)."
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
                    methods = getattr(m, "supported_generation_methods", []) or []
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
                "Analiziraj sljedeći isječak web stranice i iznesi najvažnije, konkretne i unikatne informacije vezane isključivo uz sadržaj. "
                "Izvuci činjenice koje donose jasnu vrijednost korisniku – npr. ključne novosti, specifične podatke, raritetne detalje, imena, datume, cijene, novo ili neočekivano u tekstu. "
                "Odgovori jasno i sažeto koristeći 3 do 7 bullet točaka na hrvatskom jeziku. "
                "Nemoj navoditi općenite informacije, izbjegavaj uvodne/zaključne rečenice, JSON, code blokove i parafraziranje već poznatih opisa stranice. "
                "Navedi isključivo ono što je unikatno, korisno i specifično za ovaj isječak.\n\n"
                "Isječak:\n" + excerpt
            )
            last_err = None
            for model_name in candidate_models:
                try:
                    if not model_name:
                        continue
                    model = genai.GenerativeModel(model_name)
                    resp = model.generate_content(prompt)
                    text = getattr(resp, "text", "") or "(prazan odgovor)"
                    # If model still returned JSON, try to prettify to human bullets
                    try:
                        import json

                        if text.strip().startswith("{") or text.strip().startswith("["):
                            data = json.loads(text)
                            if isinstance(data, dict):
                                lines = []
                                for k, v in data.items():
                                    lines.append(f"- {k}: {v}")
                                return "\n".join(lines)
                            if isinstance(data, list):
                                return "\n".join(f"- {item}" for item in data)
                    except Exception:
                        pass
                    return text
                except Exception as inner_e:
                    last_err = inner_e
                    continue
            if last_err is not None:
                return f"ERROR from Gemini: {last_err}"
            return "(prazan odgovor)"
        except Exception as e:
            return f"ERROR from Gemini: {e}"

    view = _mo.md("Unesite URL i kliknite ‘Scrape & summarize’.")
    if run_btn.value:
        url = (url_input.value or "").strip()
        if not url:
            view = _mo.md("Please enter a URL.")
        elif not steel_key or not google_key:
            view = _mo.md("Missing STEEL_API_KEY or GOOGLE_API_KEY in .env")
        else:
            steel_text = steel_fetch_text(url, steel_key)
            if not isinstance(steel_text, str):
                steel_text = str(steel_text)
            excerpt = (steel_text or "").strip().replace("\n", " ")[:500]
            llm_out = gemini_summarize(excerpt, google_key) if excerpt else "(nema teksta)"
            view = _mo.md(
                f"**Scraped URL:** {url}\n\n"
                f"**Steel output (isječak, ~500 znakova):**\n\n{excerpt}\n\n"
                f"**LLM rezultat:**\n\n{llm_out}"
            )

    view


if __name__ == "__main__":
    app.run()

