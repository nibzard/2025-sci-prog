import marimo

__generated_with = "0.16.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import os
    import requests
    from bs4 import BeautifulSoup
    import google.generativeai as genai

    import os
    import requests
    return BeautifulSoup, genai, os, requests


@app.cell
def _(os):
    from pathlib import Path
    from dotenv import load_dotenv

    env_path = Path.cwd() / ".env"

    load_dotenv(dotenv_path=env_path, override=True)

    STEEL_API_KEY=os.getenv("STEEL_API_KEY", "")
    GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY", "")

    TARGET_URL = "https://www.pmfst.unist.hr/"

    print(f"Steel API Key present: {bool(STEEL_API_KEY)}")
    print(f"Google API Key present: {bool(GOOGLE_API_KEY)}")
    print(f"Target URL: {TARGET_URL}")
    return GOOGLE_API_KEY, STEEL_API_KEY, TARGET_URL


@app.cell
def _(
    BeautifulSoup,
    GOOGLE_API_KEY,
    STEEL_API_KEY,
    TARGET_URL,
    genai,
    requests,
):

    # 2. Funkcija za dohvat teksta preko Steel API
    def steel_fetch_text(url: str, api_key: str) -> str:
        if not api_key:
            return "ERROR: STEEL_API_KEY nije postavljen."
        try:
            resp = requests.post(
                "https://api.steel.dev/v1/scrape",
                headers={
                    "steel-api-key": api_key,
                    "content-type": "application/json",
                    "accept": "application/json",
                },
                json={
                    "url": url,
                    "extract": ["text", "html"],
                    "useProxy": False,
                    "render": True,      
                    "fullPage": True,    
                    "delay": 2,  
                    "region": ""
                },
            )

            ct = resp.headers.get("content-type", "")
            if "application/json" in ct:
                data = resp.json()
                if isinstance(data, dict):
                    for key in ("text", "content", "extracted_text"):
                        val = data.get(key)
                        if isinstance(val, str) and val.strip():
                            return val
                    content = data.get("content")
                    if isinstance(content, dict):
                        html = content.get("html") or content.get("body")
                        if isinstance(html, str) and html.strip():
                            soup = BeautifulSoup(html, "html.parser")
                            return soup.get_text(" ", strip=True)
                    html = data.get("html")
                    if isinstance(html, str) and html.strip():
                        soup = BeautifulSoup(html, "html.parser")
                        return soup.get_text(" ", strip=True)
                return str(data)
            return resp.text
        except Exception as e:
            return f"ERROR contacting Steel: {e}"

    # 3. Funkcija za analizu teksta preko Google Gemini
    def gemini_analyze_msn(text: str, api_key: str) -> str:
        if not api_key:
            return "ERROR: GOOGLE_API_KEY nije postavljen."
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-2.5-flash")

            prompt = (
                "Analiziraj sljedeći tekst sa naslovne stranice fakulteta PMF-a u Splitu i izvuci TOP 5 najvažnijih vijesti.\n\n"
                "Za svaku vijest i diplomski studij navedi:\n"
                "- Naslov vijesti\n"
                "- Kratki opis (1 rečenica)\n\n"
                "Na kraju napiši sažetak (2-3 rečenice): koje su glavne teme danas na PMF-u?\n\n"
                "Formatiraj odgovor kao listu s bullet točkama na hrvatskom jeziku.\n\n"
                "Nemoj koristiti JSON ili code blokove, samo čisti tekst.\n\n"
                f"Tekst:\n{text[:8000]}"
            )

            resp = model.generate_content(prompt)
            return resp.text
        except Exception as e:
            return f"ERROR from Gemini: {e}"

    # 4. Pokreni scraping i analizu
    steel_text = steel_fetch_text(TARGET_URL, STEEL_API_KEY)
    print(f"\n✅ Steel output length: {len(steel_text)} znakova")

    if not steel_text.startswith("ERROR") and steel_text.strip():
        excerpt = steel_text[:800]
        print("\n=== Steel Output (isječak) ===")
        print(excerpt)

        llm_out = gemini_analyze_msn(steel_text, GOOGLE_API_KEY)
        print("\n=== 🤖 Gemini Analiza ===")
        print(llm_out)
    else:
        print(f"❌ {steel_text}")

    return


if __name__ == "__main__":
    app.run()
