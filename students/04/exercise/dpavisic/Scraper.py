import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import os
    import requests
    from pathlib import Path
    from dotenv import load_dotenv

    script_folder = Path(__file__).parent
    dotenv_path = script_folder / ".env"
    load_dotenv(dotenv_path=dotenv_path)

    STEEL_API_KEY = os.getenv("STEEL_API_KEY")

    url = "https://store.playstation.com/en-hr/pages/latest"

    endpoint = "https://api.steel.dev/v1/scrape"

    payload = {
        "url": url,
        "format": ["html"],
        "screenshot": False,
        "pdf": False,
        "delay": 1,
        "useProxy": False,
        "region": ""
    }

    headers = {
        "Content-Type": "application/json",
        "steel-api-key": STEEL_API_KEY
    }

    response = requests.post(endpoint, headers=headers, json=payload)

    if response.status_code == 200:
        data = response.json()
        raw_html = data["content"]["html"]
    else:
        print(response.text)
    return os, raw_html


@app.cell
def _(raw_html):
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(raw_html, "html.parser")

    featured_and_recommended_html = soup.find("section", {"class": "ems-sdk-strand"})

    if not(featured_and_recommended_html):
        print("Not found")
    return (featured_and_recommended_html,)


@app.cell
def _(featured_and_recommended_html, os):
    from google import genai
    from google.genai import types

    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

    client = genai.Client()
    ai_response = client.models.generate_content(
        model="gemini-2.0-flash",
        config=types.GenerateContentConfig(
            system_instruction=[
                str(featured_and_recommended_html), 
                "You're a huge Nintendo nerd, and you especially like Pokemon."
            ]
        ),
        contents="Tell me the news in Nintendo games \
        and all about the new Pokemon games \
        as well as their price based on the provided data. \
        Do not use Markdown-specific characters such \
        as asterisks and escape characters or emojis."
    )

    print(ai_response.text)
    return


if __name__ == "__main__":
    app.run()