# Opis projekta

URL: https://downdetector.com/status/cloudflare/

- Steel browser API scrape-a sadržaj stranice koja prati status Cloudflare-ovih usluga.
- Gemini analizira sentiment u komentarima i vraća 3 komentara koji najbolje obuhvaćaju generalni sentiment korisnika.

# Kako pokrenuti

- Dodati .env file sa STEEL_API_KEY i GOOGLE_API_KEY
- `pip install -r requirements.txt`
- `python scraper.py`