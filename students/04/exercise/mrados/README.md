# PMFST Web Scraper & Analyzer

Ovaj projekt demonstrira korištenje **Steel.dev API-ja** za web scraping i **Google Gemini API-ja** za analizu i sažimanje sadržaja.  
Scrapamo naslovnu stranicu **https://www.pmfst.unist.hr/** iz razloga vježbe i znatiželje – cilj je vidjeti kako se struktura stranice parsira, dohvatiti osnovne informacije (studiji, obavijesti, vijesti) i zatim ih sažeti pomoću AI modela.

## Razlog
Projekt je zamišljen kao **vježba u web scrapanju i obradi teksta**.  
Kroz njega se uči:
- kako dohvatiti HTML i tekstualni sadržaj stranice fakulteta,
- kako se nositi s dinamičkim sadržajem,
- kako koristiti LLM (Google Gemini) za generiranje sažetaka i izdvajanje ključnih informacija.

## Pokretanje
Skriptu pokrećeš jednostavno iz terminala:

```bash
python main.py
