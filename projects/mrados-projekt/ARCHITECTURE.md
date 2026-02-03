# LaptopAI – Arhitektura sustava

Ovaj dokument objašnjava unutarnju strukturu LaptopAI sustava, tok podataka i tehničke odluke.

---

## Pregled sustava

```
┌─────────────────┐
│  Korisnik       │
│  (Web UI)       │
└────────┬────────┘
         │
         v
┌─────────────────────────────────────────────────────┐
│              FastAPI Backend                        │
│  ┌──────────────────────────────────────────────┐  │
│  │  /api/compare endpoint                       │  │
│  │  - Prima 2 laptop imena                      │  │
│  │  - Paralelna analiza                         │  │
│  │  - Caching layer (koristi postojeće JSONove) │  │
│  └──────────────────┬───────────────────────────┘  │
└────────────────────┼──────────────────────────────┘
                     │
                     v
┌────────────────────────────────────────────────────┐
│         Data Collection Layer (Scraping)           │
│  ┌─────────────────────────────────────────────┐  │
│  │  Reddit Search Scraper                      │  │
│  │  - Pronalazi relevantne postove             │  │
│  │  - Ekstrahira title, body, comments         │  │
│  │  - Sprema u data/reddit/*.json              │  │
│  └─────────────────────────────────────────────┘  │
└────────────────────┬───────────────────────────────┘
                     │
                     v
┌────────────────────────────────────────────────────┐
│        Knowledge Storage Layer (ChromaDB)          │
│  ┌─────────────────────────────────────────────┐  │
│  │  Vector Embeddings                          │  │
│  │  - SentenceTransformers (all-MiniLM-L6-v2)  │  │
│  │  - Semantičko pretraživanje                 │  │
│  │  - Metapodaci (URL, timestamp, subreddit)   │  │
│  └─────────────────────────────────────────────┘  │
└────────────────────┬───────────────────────────────┘
                     │
                     v
┌────────────────────────────────────────────────────┐
│           Analysis Layer (LLM)                     │
│  ┌─────────────────────────────────────────────┐  │
│  │  Google Gemini API                          │  │
│  │  - Retrieval Augmented Generation (RAG)     │  │
│  │  - Ekstrakcija pros/cons                    │  │
│  │  - Sentiment score (1-100)                  │  │
│  │  - Korisnička preporuka                     │  │
│  └─────────────────────────────────────────────┘  │
└────────────────────┬───────────────────────────────┘
                     │
                     v
┌────────────────────────────────────────────────────┐
│            Output Layer (JSON Results)             │
│  ┌─────────────────────────────────────────────┐  │
│  │  analysis/<laptop-name>.json                │  │
│  │  - Strukturirani rezultati analize          │  │
│  │  - Cacheirani za buduće upite               │  │
│  └─────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────┘
```

---

## Arhitektura po slojevima

### 1. Data Collection Layer

**Svrha:** Prikupljanje sirovih korisničkih recenzija s Reddita.

**Komponente:**
- `scrape_query_reddit.py` – pronalazi relevantne postove putem HTML scrapinga Reddit search stranica
- `reddit_post_scrapper.py` – ekstrahira sadržaj individualnih postova i komentara
- `pipeline.py` – orkestrira cijeli proces (search → scrape → embed → analyze)

**Tehnologija:**
- **BeautifulSoup4** – parsiranje HTML-a
- **Requests** – HTTP zahtjevi
- **urllib.parse** – encoding search parametara

**Kako radi search scraping (`scrape_query_reddit.py`):**
1. Prima laptop ime (npr. "Lenovo Legion Y540")
2. Generira search queries: `["lenovo legion y540", "lenovo legion y540 review", ...]`
3. Konstruira Reddit search URL: `https://www.reddit.com/search/?q=<encoded_query>`
4. Šalje HTTP GET zahtjev s `User-Agent` headerom (imitacija browsera)
5. BeautifulSoup parsira HTML response
6. Ekstrahiraju se linkovi koji sadrže `/comments/` (postovi)
7. Filtrira duplikati pomoću `set()`
8. Sprema se u `data/reddit/<laptop-slug>/search_results.json`

**Kako radi post scraping (`reddit_post_scrapper.py`):**
1. Prima URL reddit posta (dobiven iz search scrapera)
2. Fetch-a HTML stranicu posta
3. BeautifulSoup pronalazi:
   - `<h1>` tag → naslov posta
   - `<div property="schema:articleBody">` → tijelo posta
   - `<div slot="comment">` → top-level komentare
4. Ekstrahira tekst iz svih `<p>` paragrafa
5. Vraća strukturirani dict:
   ```python
   {
       "url": str,
       "title": str,
       "body": str,
       "comments": list[str],  # prvih 10
       "scraped_at": str
   }
   ```

**Izlaz:**
- `data/reddit/<laptop-name>/*.json` – strukurirani JSON s podacima o postovima

**Ključne odluke:**
- Koristi se **search-based scraping** umjesto front-page scrapinga jer:
  - Omogućava pronalazak starijih postova
  - Povećava relevantnost rezultata
  - Skalira se na bilo koji laptop model
- **Zašto ne PRAW (Reddit API)?**
  - Reddit je odbio odobriti pristup API-ju
  - HTML scraping je jedina opcija bez OAuth autentifikacije
  - Funkcionira i bez API ključeva

---

### 2. Knowledge Storage Layer

**Svrha:** Semantičko pohranjivanje znanja za brzo pretraživanje.

**Tehnologija:** ChromaDB (lokalna vector baza)

**Komponente:**
- `vectorstore.py` – embediranje i pohranjivanje dokumenata
- Embeddinzi generiraju se pomoću `sentence-transformers/all-MiniLM-L6-v2`

**Što se pohranjuje:**
- Tekst postova i komentara
- Metapodaci (URL, subreddit, timestamp)
- Vector embeddinzi (384-dimenzionalni vektori)

**Zašto ChromaDB?**
- ✅ Lokalno – bez dodatnih servisa
- ✅ Brzo semantičko pretraživanje
- ✅ Persistent storage
- ✅ Skalabilno – lako se dodaju novi laptopi

---

### 3. Analysis Layer

**Svrha:** Generiranje strukturiranih insights pomoću AI-a.

**LLM:** Google Gemini 1.5 Pro

**Proces:**
1. Query ChromaDB s laptop imenom
2. Dohvati relevantne embeddinze (top 10-20 rezultata)
3. Konstruiraj prompt s kontekstom
4. Šalji LLM-u
5. Parsiraj JSON odgovor

**Output format:**
```json
{
  "sentiment_score": 78,
  "pros": ["prednost 1", "prednost 2", ...],
  "cons": ["nedostatak 1", "nedostatak 2", ...],
  "key_themes": ["performance", "build quality", ...],
  "sentiment_explanation": "Korisnici su zadovoljni...",
  "user_recommendation": "Preporučeno za gaming..."
}
```

**Zašto Gemini?**
- Dobar JSON parsing
- Veliki context window (može primiti više postova)
- Dobra sentiment analiza na korisničkom jeziku

---

### 4. API Layer (FastAPI)

**Svrha:** REST API za frontend i buduće integracije.

**Endpointi:**

#### `POST /api/compare`
```json
{
  "laptop1": "Lenovo Legion Y540",
  "laptop2": "Dell XPS 15"
}
```

**Logika:**
1. **Caching check** – provjerava postoje li već generirani JSONovi u `analysis/`
2. **Paralelna analiza** – ako cache ne postoji, pokreće `run_single_analysis()` za oba laptopa istovremeno
3. **Usporedba** – uspoređuje sentiment_score i vraća winnera
4. **Response** – JSON s rezultatima oba laptopa

**Zašto caching?**
- LLM pozivi su skupi i spori
- Većina upita je za iste popularne laptope
- Analiza se ne mijenja često (osim ako se ne dodaju novi postovi)

---

## Tok podataka

### Puni pipeline (prvi put za laptop)

```
1. Korisnik: "Lenovo Legion Y540"
   ↓
2. Reddit Search Scraper
   → pronalazi ~50 relevantnih postova
   ↓
3. Reddit Post Scraper
   → ekstrahira title, body, komentare
   → sprema u data/reddit/lenovo-legion-y540/*.json
   ↓
4. Markdown Generator
   → generira čitljive .md datoteke (opcionalno)
   ↓
5. ChromaDB Embedding
   → embedira postove
   → sprema u ./chroma/
   ↓
6. LLM Analiza (Gemini)
   → dohvaća relevantne postove iz ChromaDB
   → generira pros/cons i sentiment
   → sprema u analysis/lenovo_legion_y540.json
   ↓
7. API Response
   → vraća JSON frontend-u
```

### Cached pipeline (ponovljeni upit)

```
1. Korisnik: "Lenovo Legion Y540"
   ↓
2. API provjerava: analysis/lenovo_legion_y540.json
   → Postoji ✅
   ↓
3. API Response
   → učitava JSON i vraća ga odmah
   → <200ms umjesto ~10s
```

---

## Rukovanje greškama

### API Failures
- Reddit scraper retry logika (3 pokušaja)
- Gemini API timeout handling (30s)
- Graceful degradation (vraća se prazan rezultat s error flagom)

### Prazni rezultati
- Ako ChromaDB ne pronađe dovoljno postova → vraća se upozorenje
- Ako LLM vrati nevaljani JSON → parsira se partially ili vraća error

### Unexpected LLM output
- Validacija JSON strukture prije spremanja
- Fallback na default vrijednosti ako su ključevi nedostajući

---

## Dizajnerske odluke

### Zašto HTML scraping umjesto Reddit API?

**Problem:**
- Reddit službeni API (PRAW) zahtijeva OAuth autentifikaciju
- Reddit je odbio odobriti pristup za ovaj projekt
- `.json` endpointi su blokirani za cloud IP adrese (npr. Google Colab)

**Rješenje:**
- **BeautifulSoup + Requests** za HTML scraping
- Imitacija normalnog browsera pomoću `User-Agent` headera
- Parsing HTML strukture umjesto JSON API-ja

**Prednosti:**
- ✅ Ne zahtijeva API ključeve
- ✅ Ne zahtijeva OAuth autentifikaciju
- ✅ Radi iz bilo kojeg okruženja (local, cloud)

**Nedostaci:**
- ⚠️ Krhko – mijenja se ako Reddit promijeni HTML strukturu
- ⚠️ Sporije od API-ja
- ⚠️ Potencijalno blokiranje ako se šalje previše requesta

**Mitigacija:**
- `time.sleep(2)` između requesta kako bi se izbjeglo rate limiting
- Error handling za HTTP 403/429 odgovore
- User-Agent rotation (buduća implementacija)

---

### Zašto ChromaDB umjesto SQL baze?

| ChromaDB | SQL |
|----------|-----|
| Semantičko pretraživanje (keyword independence) | Exact match (zahtjeva iste riječi) |
| Automatski embeddinzi | Ručno indeksiranje potrebno |
| Skalira s velikim tekstovima | Loše za long-form content |
| RAG-friendly | Kompleksna integracija s LLM-om |

### Zašto modularni pipeline?

- **Testabilnost** – svaki modul se može testirati zasebno
- **Skalabilnost** – lako se dodaju novi sources (YouTube, forums)
- **Debugging** – jasno je gdje je problem ako nešto ne radi
- **Reusability** – scraper može raditi bez LLM-a, LLM bez scrapera

### Zašto generirani podaci nisu commitani?

- `data/`, `chroma/`, `analysis/*.json` su u `.gitignore` jer:
  - Veliki fileovi (ChromaDB može biti >100MB)
  - Često se mijenjaju (svaki run pipeline-a)
  - Reproducibilni (može se regenerirati pokretanjem `pipeline.py`)
  - Sadržaj može biti copyright-protected (Reddit postovi)

### Zašto JSON output?

- Frontend-friendly (direktno se parsira u JavaScriptu)
- Human-readable (lako debuganje)
- Strukturirano (tipovi podataka su jasni)
- LLM-friendly (Gemini nativno podržava JSON mode)

---

## Ograničenja

### Ovisnost o kvaliteti Reddit podataka
- Ako nema dovoljno postova → loša analiza
- Ako su postovi stari (>2 godine) → možda nije relevantno
- Ako su korisnici pristrani (fanboys) → iskrivljen sentiment

### AI output varijabilnost
- LLM ne vraća uvijek iste rezultate (non-determinizam)
- Ponekad vraća previše generične odgovore
- Sentiment score nije egzaktna znanost

### Nema real-time scrapinga
- Pipeline se pokreće ručno
- Novi postovi se ne prikupljaju automatski
- Za osvježavanje potrebno ponovno pokrenuti `pipeline.py`

### Rate limiting
- Reddit može blokirati previše requesta
- Gemini API ima dnevna ograničenja (free tier)

---

## Buduća poboljšanja arhitekture

### 1. Web Frontend Deployment
- Docker kontejnerizacija (FastAPI + React)
- Nginx reverse proxy
- Deployment na Render/Railway/Vercel

### 2. Scheduled Scraping
- Cron job za dnevno osvježavanje
- Background tasks (Celery)
- Inkrementalno dodavanje novih postova

### 3. Multi-Source Input
- YouTube transkripti (API ili scraping)
- Tech forumi (NotebookCheck, LaptopMedia)
- Amazon recenzije

### 4. Advanced Caching Layer
- Redis za brži cache
- Invalidacija cachea nakon N dana
- Precomputed comparisons (cache parova laptopa)

### 5. Better Error Handling
- Retry logika s exponential backoff
- Monitoring (Sentry)
- User-friendly error poruke

### 6. Performance Optimizations
- Batching LLM requesta
- Parallel embedding generation
- Smanjenje ChromaDB query vremena

---

## Zaključak

LaptopAI koristi modernu RAG (Retrieval Augmented Generation) arhitekturu koja kombinira:
- Scraping za prikupljanje znanja
- Vector bazu za brzo pretraživanje
- LLM za generiranje strukturiranih insights
- REST API za integraciju s frontend-om

Sustav je dizajniran modularno, što omogućava lagano proširivanje i održavanje.
