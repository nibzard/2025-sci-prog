# Simple Web Scraper (Steel + Google AI)

## Setup

1. Install dependencies:

   ```bash
   python3 -m pip install --user marimo requests google-generativeai python-dotenv
   ```

2. Copy the sample environment file and add your API keys:
   ```bash
   cp students/04/exercise/dbaric/env.sample students/04/exercise/dbaric/.env
   ```
   Edit `.env` and fill in:
   ```
   STEEL_API_KEY=your_steel_key_here
   GOOGLE_API_KEY=your_google_key_here
   ```

## How to Run

```bash
python3 -m marimo run students/04/exercise/dbaric/scraper.py
```

- Enter a URL in the UI
- Click "Scrape & summarize"

## Output

- Scraped URL
- Excerpt from Steel (~500 characters)
- Gemini LLM summary

_Tip:_ If you get Steel 401/404 errors, check your API key or endpoint.
