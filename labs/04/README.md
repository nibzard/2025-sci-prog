## Cilj zadatka

1. Napraviti **minimalni scraper** neke javno dostupne web stranice po vlastitom izboru (npr. portal vijesti, stranica kolegija, stranica proizvoda, sportski rezultati…).
2. Scraper mora koristiti:

   * **Steel API** za dohvat sadržaja stranice (tekst / HTML / image).
   * **Google AI API** za analizu tog sadržaja (sažetak, izdvajanje podataka).
3. Sve to treba izvesti u **Marimo notebooku** (Python).
4. U Pull Request poruku (description) zalijepiti:

   * URL stranice koju ste scrapali
   * dio sirovog teksta koji je vratio Steel
   * rezultat obrade (sažetak / strukturirani podaci) koji je vratio Google AI API.

👉 Ideja: pokažite da znate dohvatiti web i onda "pročitati" i interpretirati taj sadržaj pomoću modela.

---

## Primjeri što zadatak može biti (inspiracija)

Možete birati sami, ovo su samo smjerovi:

### 1. Vijesti

* Scrapeati naslovnu stranicu nekog news portala.
* Izvući 5 najvažnijih naslova dana (naslov + podnaslov).
* Pomoću Google AI API-ja:

  * generirati kratki sažetak situacije (“što se danas dogodilo?”),
  * ili klasificirati vijesti po temama (politika / sport / tech / zabava).

Što ide u PR:

* "Scraped URL: ..."
* "Top 5 vijesti (raw text iz Steel-a): ..."
* "LLM sažetak dana: ..."

---

### 2. Stranica kolegija / objava s fakulteta

* Scrapeati obavijesti s weba odsjeka / katedre / kolegija.
* Pomoću Google AI API-ja:

  * izdvojiti rokove (datumi ispita, rok predaje projekta),
  * prebaciti to u listu bullet točaka "Što moram zapamtiti ove sedmice".

Što ide u PR:

* "Scraped URL: ..."
* "Izdvojeni datumi i rokovi (LLM): ..."
* "Napomena: Rok za projekt je xx.xx.2025. u yy:yy"

---

### 3. Proizvodi / cijene

* Scrapeati stranicu nekog webshopa (npr. lista laptopa, grafičkih kartica, knjiga).
* Pomoću Google AI API-ja:

  * izvući naziv proizvoda, trenutnu cijenu i eventualno dostupnost,
  * sortirati po cijeni i sažeti ("najjeftiniji model je ...", "srednja cijena je ...").

Što ide u PR:

* "Scraped URL: ..."
* "Tablica proizvoda (LLM ekstrakcija): ime / cijena / dostupnost"

(Napomena: ne dirati login zone ni paywall. Samo javno dostupne stranice.)

---

### 4. Sport

* Scrapeati stranicu sa rezultatima ili rasporedom utakmica.
* Pomoću Google AI API-ja:

  * napraviti kratak pregled (“Tko je pobijedio?”, “Kada je sljedeća utakmica?”, “Koji su ključni igrači spomenuti?”),
  * ili generirati bullet listu s datumima i protivnicima.

Što ide u PR:

* "Scraped URL: ..."
* "Rezultati (raw excerpt): ..."
* "Sažetak sportskog kola (LLM): ..."

---

### 5. FAQ / dokumentacija / uvjeti korištenja

* Scrapeati neku stranicu s pravilima (npr. pravila privatnosti, FAQ servisa, opis tarife operatera).
* Pomoću Google AI API-ja:

  * prevesti ključne točke na hrvatski jednostavnim jezikom (“Što oni zapravo smiju raditi s mojim podacima?”),
  * ili izvući samo bitne točke ("Što se naplaćuje?", "Koja su ograničenja?").

Što ide u PR:

* "Scraped URL: ..."
* "Izvorni tekst (prvih 500 znakova): ..."
* "LLM objašnjenje za ljude: ..."

---

## Što konkretno predajete

U vašem projektu / folderu:

1. `scraper.py` (marimo)

   * Kod koji:

     * pozove Steel API za jedan URL,
     * uzme dobiveni tekst,
     * pošalje taj tekst u Google AI API,
     * ispiše rezultate koje ćete kopirati u PR.
2. `README.md`

   * kratko objašnjenje:

     * koji URL scrapate i zašto,
     * kako pokrenuti skriptu (python komanda),
     * koje environment varijable treba postaviti (`STEEL_API_KEY`, `GOOGLE_API_KEY`).

U Pull Request opisu (obavezno copy/paste outputa iz marimo-a):

* **Scraped URL:** (točan URL)
* **Steel output (isječak):** prvih ~300-500 znakova čistog teksta koji ste dobili sa stranice
* **LLM rezultat:** sažetak / tablica informacija / bullet liste rokova / itd.

Time se vidi da:

1. Steel radi (dohvaća web).
2. Google AI API radi (analizira sadržaj).
3. Vi znate cijeli put od web → podaci → zaključak.




