# IMDb Top 10 Movies Scraper (Marimo + Steel API + Gemini AI)

Ovaj projekt scrapa IMDb listu **Most Popular Movies**, parsira HTML pomoću
BeautifulSoup i zatim koristi **Gemini 2.0 Flash** model za generiranje
detaljne analize TOP 10 filmova:

- naslov
- godina
- žanr
- kratki opis
- popularnost / trending razlozi
- IMDb ocjena (ako se može izvući)
- preporuka za gledanje
- stroga rang lista 1–10

Aplikacija radi kao **Marimo notebook**.

---

### 1. Instalacija paketa

U Codespaceu / terminalu pokrenuti:

```bash
pip install marimo python-dotenv requests beautifulsoup4 google-genai

### Kreirati .env datoteku

u fileu .env upisi podatke za:
STEEL_API_KEY=ovdje_upisi_svoj_steel_api_key
GEMINI_API_KEY=ovdje_upisi_svoj_gemini_api_key
