# LaptopAI – Analiza korisničkih recenzija laptopa pomoću AI-a

LaptopAI automatski prikuplja Reddit recenzije laptopa, semantički ih pohranjuje i koristi AI za generiranje strukturiranih preporuka temeljenih na sentimentu korisnika.

---

## Problem

Kupnja laptopa je otežana zbog:
- **Previše rasutih recenzija** – korisnici dijele iskustva na desetke subreddita
- **Nemoguće usporedbe** – teško je usporediti iskustva različitih korisnika
- **Gubljen vremena** – nitko ne želi čitati stotine postova i komentara
- **Konfuzne informacije** – često su službene recenzije nedovoljne ili pristrane

---

## Rješenje

LaptopAI rješava ovaj problem kroz automatizirani pipeline:

1. **Prikupljanje** – pronalazi relevantne Reddit rasprave o laptopima
2. **Pohranjivanje** – sprema ih u semantički pretraživu bazu znanja
3. **Analiza** – koristi AI za ekstrakciju sentimenta, prednosti i nedostataka
4. **Rezultat** – isporučuje čistu, strukturiranu preporuku i usporedbu

---

## Ključne funkcionalnosti

🔍 **Reddit scraping** – automatsko prikupljanje korisničkih recenzija  
🧠 **Semantičko pretraživanje** – embeddinzi omogućuju pronalaženje relevantnih informacija  
🤖 **AI analiza sentimenta** – Google Gemini ekstrahira pros/cons i ocjenjuje laptope  
📊 **Strukturirani output** – JSON rezultati spremni za frontend  
⚔️ **Laptop Battle UI** – web sučelje za usporedbu dva laptopa u realnom vremenu  
📁 **Modularni dizajn** – nezavisni scraper, vector store i LLM slojevi  

---

## Primjer rezultata

### Unos
```
Laptop 1: Lenovo Legion Y540
Laptop 2: Dell XPS 15
```

### Izlaz
```json
{
  "laptop_name": "Lenovo Legion Y540",
  "sentiment_score": 78,
  "pros": [
    "Odličan omjer cijene i performansi",
    "Dobro hlađenje uz RTX 2060",
    "Kvalitetna tipkovnica"
  ],
  "cons": [
    "Loša baterija (2-3 sata)",
    "Osrednji ekran (sRGB ~60%)",
    "Plastični build quality"
  ],
  "user_recommendation": "Preporučeno za gaming na budžetu, ali ne za profesionalnu upotrebu."
}
```

---

## Tehnologije

**Backend**
- Python 3.11+
- FastAPI (REST API)
- ChromaDB (vector baza podataka)
- SentenceTransformers (embeddinzi)
- Google Gemini API (LLM analiza)

**Frontend**
- React 18
- Vite
- TailwindCSS
- Axios

**Scraping**
- BeautifulSoup4
- Requests

---

## Kako pokrenuti projekt

### 1. Backend (FastAPI)

```bash
# Instalacija dependencies
pip install -r requirements.txt

# Pokretanje API servera
python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

API će biti dostupan na `http://localhost:8000`

### 2. Frontend (React)

```bash
# Ulazak u frontend folder
cd laptop-battle-ui

# Instalacija dependencies
npm install

# Pokretanje development servera
npm run dev
```

Frontend će biti dostupan na `http://localhost:5173`

### 3. Pipeline (opcionalno – za scraping novih laptopa)

```bash
python pipeline.py
```

---

## Primjer API poziva

```bash
POST http://localhost:8000/api/compare
Content-Type: application/json

{
  "laptop1": "Lenovo Legion Y540",
  "laptop2": "Dell XPS 15"
}
```

**Odgovor:**
```json
{
  "laptop1": { ... },
  "laptop2": { ... },
  "winner": "laptop1",
  "comparison_summary": "Lenovo Legion Y540 pruža bolje gaming performanse uz nižu cijenu..."
}
```

---

## Status projekta

✅ Reddit scraping pipeline  
✅ ChromaDB vector storage  
✅ LLM sentiment analiza  
✅ FastAPI backend s caching sustavom  
✅ React frontend s battle UI  
✅ Usporedba dva laptopa  

---

## Budući razvoj

🔮 **Više izvora podataka** – dodavanje YouTube transkripata, foruma, tech blogova  
🔮 **Automatski scheduled scraping** – dnevno osvježavanje baze znanja  
🔮 **Historijski tracking** – praćenje promjena sentimenta kroz vrijeme  
🔮 **Napredne usporedbe** – više od 2 laptopa, performance grafovi  
🔮 **Deployment** – Docker kontejnerizacija i hosting  

---

## Autori

Projekt razvijen u sklopu kolegija **Završni projekt** na PMF-ST.

---

## Licenca

MIT License
