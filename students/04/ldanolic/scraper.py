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

    # NEW URL → IMDb Most Popular Movies
    url = "https://www.imdb.com/chart/moviemeter/"

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

    # IMDb table with most popular movies
    movies_table = soup.find("table", {"class": "ipc-metadata-list"})

    if not movies_table:
        print("Movies table not found!")
    
    return (movies_table,)


@app.cell
def _(movies_table, os):
    from google import genai
    from google.genai import types

    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

    client = genai.Client()

    # AI PROMPT → Highly detailed analysis, WITH RANKING 1–10
    ai_response = client.models.generate_content(
        model="gemini-2.0-flash",
        config=types.GenerateContentConfig(
            system_instruction=[
                str(movies_table),
                "You are a professional movie critic with deep knowledge \
                of film history, genres, storytelling and cinematography. \
                You analyze movie lists with precision and style."
            ]
        ),
        contents=(
            "Using the provided IMDb movie list HTML, extract exactly the TOP 10 "
            "most popular movies. For each movie provide the following: "
            "1. Movie title; "
            "2. Release year; "
            "3. Genre (if possible); "
            "4. Short plot description; "
            "5. IMDb rating (if visible or inferable); "
            "6. Summary of why the movie is currently trending; "
            "7. Recommendation: who should watch it and why. "
            "IMPORTANT: Present your answer **strictly as a ranking list from 1 to 10**, "
            "where each line begins with the rank number (1., 2., ..., 10.) followed by the movie information. "
            "Write everything in plain text (no markdown)."
        ),
    )

    print(ai_response.text)
    return


if __name__ == "__main__":
    app.run()
