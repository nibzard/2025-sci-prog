# Hacker News Scraper

## Opis projekta

Ovaj projekt scrapa [Hacker News](https://news.ycombinator.com/) naslovnu stranicu i koristi Google AI (Gemini) za analizu i ekstrakciju top 10 najvažnijih vijesti dana.

---

## Koji URL scrapamo i zašto?

**URL:** `https://news.ycombinator.com/`

**Zašto Hacker News?**

- **Javno dostupan** - ne zahtijeva login ni registraciju
- **Bogat sadržaj** - tech vijesti, startups, programiranje, inovacije
- **Dinamički** - sadržaj se konstantno mijenja, uvijek ima novih vijesti
- **Strukturiran** - jasno definirani naslovi, bodovi (points) i komentari
- **Relevantno** - jedna od najpoznatijih tech zajednica na internetu
- **Idealno za demo** - pokazuje kako kombinirati web scraping i AI analizu

### Što scraper radi?

1. **Steel API** - Dohvaća čisti tekst sa Hacker News naslovne stranice
2. **Google AI (Gemini)** - Analizira tekst i izvlači:
   - Top 10 najvažnijih vijesti
   - Naslov, broj bodova (points) i broj komentara za svaku vijest
   - Kratki opis svake vijesti (o čemu se radi)
   - Sažetak glavnih tema dana na Hacker News

---

## Instalacija

### 1. Instaliraj potrebne biblioteke

```bash
pip install marimo requests python-dotenv google-generativeai beautifulsoup4
```

### 2. Nabavi API ključeve

#### Steel API:
1. Registriraj se na [steel.dev](https://steel.dev)
2. Kreiraj novi API ključ
3. Kopiraj ga (počinje sa `ste-...`)

#### Google AI API:
1. Idi na [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Klikni "Create API Key"
3. Kopiraj ključ (počinje sa `AIzaSy...`)

---

## Pokretanje

### Pokreni Marimo notebook:

```bash
marimo edit scraper.py
```

Ovo će otvoriti Marimo notebook u browseru.

---

## ⚠️ VAŽNO: Unos API ključeva

### Zbog problema s učitavanjem iz `.env` datoteke:

Iz nekog razloga, kod **uvijek uspješno pronalazi Steel API ključ** iz `.env` datoteke, ali **ne pronalazi Google API ključ**. Čak i kada su oba ključa ispravno postavljena u `.env` datoteci, Google ključ se ne učitava.

### ✅ RJEŠENJE: Ručni unos kroz notebook

Nakon pokretanja `marimo edit scraper.py`, na vrhu stranice ćeš vidjeti **dva input polja**:

1. **Steel API Key** - (može se automatski učitati iz `.env` ili ručno unijeti)
2. **Google AI API Key** - (⚠️ **obavezno ručno unijeti**)

**Postupak:**
1. Otvori notebook: `marimo edit scraper.py`
2. Na vrhu stranice **unesi oba API ključa** u input polja:
   - Steel API Key: `ste-tvoj_ključ`
   - Google AI API Key: `AIzaSy-tvoj_ključ`
3. Scroll dolje do URL polja
4. Klikni gumb **"Scrape & Analyze HN"**
5. Pričekaj rezultate

**Napomena:** Čak i uz `.env` datoteku, **Google API ključ se neće učitati**, pa ga moraš ručno unijeti u notebook.