import marimo

__generated_with = "0.18.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import os
    import requests
    from dotenv import load_dotenv
    import google.generativeai as genai
    from bs4 import BeautifulSoup
    import json
    from pathlib import Path
    return BeautifulSoup, Path, genai, json, load_dotenv, os, requests


@app.cell
def _(Path, load_dotenv, os):
    load_dotenv()
    try:
        load_dotenv(dotenv_path=Path(
            __file__).with_name(".env"), override=False)
    except Exception:
        pass

    STEEL_API_KEY = os.getenv("STEEL_API_KEY", "")
    GOOGLE_API_KEY = os.getenv(
        "GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY", "")
    return GOOGLE_API_KEY, STEEL_API_KEY


@app.cell
def _(BeautifulSoup, json, requests):
    def scrape_with_steel(url: str, api_key: str) -> str:
        if not api_key:
            return "ERROR: Steel API key not provided.\nPlease create a .env file with: STEEL_API_KEY=your_key_here"

        print(f"🌐 Scraping {url}...")

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
                    "extract": "text",
                    "useProxy": False,
                    "delay": 2,
                    "fullPage": True,
                    "region": "",
                },
                timeout=45
            )

            if resp.status_code != 200:
                return f"ERROR: Steel API returned status {resp.status_code}: {resp.text}"

            data = resp.json()

            # Handle multiple response formats
            if isinstance(data, dict):
                # Try direct text fields
                for key in ("text", "content", "extracted_text"):
                    val = data.get(key)
                    if isinstance(val, str) and val.strip():
                        return val

                # Check nested content.html structure
                content = data.get("content")
                if isinstance(content, dict):
                    html = content.get("html") or content.get("body")
                    if isinstance(html, str) and html.strip():
                        soup = BeautifulSoup(html, "html.parser")
                        return soup.get_text(" ", strip=True)

                # Check top-level html field
                html = data.get("html")
                if isinstance(html, str) and html.strip():
                    soup = BeautifulSoup(html, "html.parser")
                    return soup.get_text(" ", strip=True)

                # Fallback: stringify JSON
                return json.dumps(data, ensure_ascii=False)

            return str(data)

        except requests.exceptions.Timeout:
            return "ERROR: Request timed out - site may be slow or unreachable"
        except requests.exceptions.RequestException as e:
            return f"ERROR: Request failed: {type(e).__name__}: {e}"
        except Exception as e:
            return f"ERROR: {type(e).__name__}: {e}"
    return (scrape_with_steel,)


@app.cell
def _(genai):
    def analyze_with_gemini(text: str, api_key: str) -> str:
        if not api_key:
            return "ERROR: Google API key not provided.\nPlease create a .env file with: GOOGLE_API_KEY=your_key_here or GEMINI_API_KEY=your_key_here."

        print("🤖 Analyzing with Gemini AI...")

        try:
            genai.configure(api_key=api_key)

            models_to_try = [
                "gemini-2.0-flash",
                "gemini-1.5-flash",
                "gemini-1.5-pro"
            ]

            prompt = f"""
    Analyze the following content scraped from DownDetector's Cloudflare status page.

    Your task:
    1. **Overall Sentiment Analysis**: Determine the general sentiment of user comments (Positive, Negative, Mixed, or Neutral). Provide a brief explanation (2-3 sentences).

    2. **Top 3 Comments**: Extract 3 user comments from the page that best describe the general sentiment. Include the actual comment text.

    3. **Key Insights**: Briefly summarize what users are reporting (are there outages, what issues are they experiencing, etc.).

    Format your response clearly with these three sections using headers and bullet points.

    Content (first 10,000 characters):
    {text[:10000]}
    """

            last_error = None
            for model_name in models_to_try:
                try:
                    print(f"  Trying model: {model_name}...")
                    model = genai.GenerativeModel(model_name)
                    response = model.generate_content(prompt)
                    return response.text
                except Exception as e:
                    last_error = e
                    print(f"  Model {model_name} failed: {type(e).__name__}")
                    continue

            return f"ERROR: All Gemini models failed. Last error: {last_error}"

        except Exception as e:
            return f"ERROR: {type(e).__name__}: {e}"
    return (analyze_with_gemini,)


@app.cell
def _(GOOGLE_API_KEY, STEEL_API_KEY, analyze_with_gemini, scrape_with_steel):
    url = "https://downdetector.com/status/cloudflare/"

    # Step 1: Scrape with Steel
    scraped_text = scrape_with_steel(url, STEEL_API_KEY)

    if scraped_text.startswith("ERROR"):
        print(f"❌ Scraping failed:\n{scraped_text}")
    else:
        print(f"✓ Successfully scraped {len(scraped_text)} characters")
        print()

        # Step 2: Display preview of scraped content
        print("=" * 70)
        print("SCRAPED CONTENT PREVIEW (First 800 characters)")
        print("=" * 70)
        excerpt = scraped_text[:800].strip()
        print(excerpt)
        if len(scraped_text) > 800:
            print("\n... [content continues] ...")
        print()
        print("=" * 70)
        print()

        # Step 3: Analyze with Gemini
        analysis = analyze_with_gemini(scraped_text, GOOGLE_API_KEY)

        if analysis.startswith("ERROR"):
            print(f"❌ AI analysis failed:\n{analysis}")

        # Step 4: Display results
        print("=" * 70)
        print("AI ANALYSIS RESULTS")
        print("=" * 70)
        print(analysis)
    return


if __name__ == "__main__":
    app.run()
