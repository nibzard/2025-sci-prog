# Plan Čišćenja Podataka

## KUĆE (houses.parquet)

### IZBRISATI (nepotrebni ili 100% None)
- `listing_id`
- `url`
- `grad` (svi su iz istog grada po datasetu)
- `vrsta_nekretnine` (sve su kuće)
- `opis`
- `image_paths`
- `scraped_at`
- `Agencijsku proviziju plaća` (92% None, a gdje postoji uvijek je "Prodavatelj")
- `Dozvole` (100% None)
- `Kat` (100% None)
- `Orijentacija stana` (100% None)
- `Tip stana` (100% None)
- `Ukupni broj katova` (100% None)
- `Ulica` (100% None)
- `Šifra objekta`
- `Namještenost i stanje` (80% None - premalo podataka)
- `Netto površina` (85% None - premalo podataka)
- `Kupaonica i WC - Broj kupaonica bez WC-a` (95% None)
- `Agencijska provizija` (96% None - premalo podataka)

---

### NUMERIČKE TRANSFORMACIJE

**cijena**
- Ukloniti " €"
- Zamijeniti "." s "" (tisućice)
- Zamijeniti "," s "." (decimale)
- Pretvoriti u float
- Primjer: "330.000 €" → 330000.0

**Stambena površina**
- Ukloniti " m²"
- Zamijeniti "." s "" (tisućice)
- Zamijeniti "," s "."
- Pretvoriti u float
- Primjer: "1.247,00 m²" → 1247.0

**Površina okućnice**
- Ista transformacija kao Stambena površina
- None ostaviti kao NaN

**Broj soba**
- Već je numerički (int), ostaviti kako je

**Broj parkirnih mjesta**
- None → NaN (ne 0!)
- Ostale vrijednosti pretvoriti u int

**Godina izgradnje**
- Ukloniti "." na kraju
- Pretvoriti u int
- None → NaN
- Dodati novi stupac: `ima_godinu_izgradnje` (1 ako ima, 0 ako NaN)

**Godina zadnje renovacije**
- Ukloniti "." na kraju
- Pretvoriti u int
- None → NaN
- Gdje je NaN, a `Godina izgradnje` postoji → kopiraj vrijednost iz `Godina izgradnje`

---

### MAPIRANJE KATEGORIJA → BROJ

**Broj etaža**
| Vrijednost | Nova vrijednost |
|------------|-----------------|
| Prizemnica | 1 |
| Visoka prizemnica | 1.5 |
| Katnica | 2 |
| Dvokatnica | 3 |
| Višekatnica | 4 |

**Energetski razred**
| Vrijednost | Nova vrijednost |
|------------|-----------------|
| A+ | 5 |
| A | 4 |
| B | 3 |
| C | 2 |
| D | 1 |
| E | 0 |
| F | -1 |
| G | -2 |
| None | NaN |

- Dodati novi stupac: `ima_energetski_razred` (1 ako ima, 0 ako NaN)

**Kupaonica i WC - Broj WC-a**
- None → NaN
- "5+" → 5
- Ostalo pretvoriti u int

**Kupaonica i WC - Broj kupaonica s WC-om**
- None → NaN
- "5+" → 5
- Ostalo pretvoriti u int

---

### BINARNO MAPIRANJE (Da/None → 1/0)

| Stupac | None → | Da → |
|--------|--------|------|
| Grijanje - Dodatni izvor grijanja | 0 | 1 |
| Grijanje - Klima uređaj | 0 | 1 |
| Pogled na more | 0 | 1 |
| Razgledavanje putem video poziva | 0 | 1 |

**Mogućnost zamjene**
- "Moguća zamjena za drugu nekretninu" → 1
- "Nije moguća zamjena za drugu nekretninu" → 0
- None → 0

---

### ONE-HOT ENCODING (multi-value stupci)

Za sve ove stupce:
- Razdvojiti po ", " (zarez + razmak)
- Kreirati stupac za svaku jedinstvenu vrijednost
- None → 0 u svim novim stupcima
- Prefiks za ime stupca = skraćeno ime originalnog stupca

**Balkon/Lođa/Terasa** → `balkon_`, `lodja_`, `terasa_`
- Jedinstvene vrijednosti: Balkon, Lođa (Loggia), Terasa
- "Nema ništa navedeno" → 0 u svima

**Dozvole i potvrde** → `dozvola_`
- Jedinstvene vrijednosti: Vlasnički list, Građevinska dozvola, Uporabna dozvola

**Funkcionalnosti i ostale karakteristike** → `funk_`
- Jedinstvene vrijednosti: Protuprovalna vrata, Alarmni sustav, Video Portafon, Kada, Tuš kabina, Jacuzzi, Sauna, Podno grijanje, Kamin, Električne rolete, Perilica posuđa, Zasebni ulaz u objekt

**Grijanje - Alternativni izvori energije** → `alt_energija_`
- Jedinstvene vrijednosti: Solarni paneli, Toplinske pumpe

**Grijanje - Sustav grijanja** → `grijanje_`
- One-hot za svaku jedinstvenu vrijednost
- None → 0 u svim stupcima
- Dodati: `ima_sustav_grijanja` flag

**Ostali objekti i površine** → `objekt_`
- Jedinstvene vrijednosti: Dvorište/vrt, Spremište/šupa, Podrum, Roštilj, Bazen, Zimski vrt, Vrtna kućica

**Podaci o objektu** → `podatak_`
- Jedinstvene vrijednosti: Novogradnja, Lift, Pristup za osobe s invaliditetom, Gradski plin, Gradski vodovod, Gradska kanalizacija

**Tip kuće** → `tip_kuce_`
- Jedinstvene vrijednosti: Samostojeća, Dvojna (duplex), U nizu, Stambeno-poslovna

**Vrsta kuće (gradnje)** → `gradnja_`
- Jedinstvene vrijednosti: Zidana kuća (beton, opeka), Kamena kuća, Montažna kuća, Drvena kuća
- None → 0 u svim stupcima (77% None, ali zadržavamo jer može biti informativan)

**Vrsta parkinga** → `parking_`
- Jedinstvene vrijednosti: Garaža, Garažno mjesto, Vanjsko natkriveno mjesto, Vanjsko ne-natkriveno mjesto, Besplatni javni parking, Naplatni javni parking

---

### LOKACIJA - RAZDVAJANJE

**Lokacija** → razdvojiti u 3 stupca:
- `zupanija` (prvi dio prije prvog zareza)
- `grad_opcina` (drugi dio)
- `naselje` (treći dio)

Ovi stupci ostaju kao kategoričke varijable (string) - LightGBM/CatBoost će ih enkodirati automatski, ili možeš koristiti label encoding.

---

## STANOVI (apartments.parquet)

### IZBRISATI (nepotrebni ili 100% None)
- `listing_id`
- `url`
- `grad`
- `vrsta_nekretnine`
- `opis`
- `image_paths`
- `scraped_at`
- `Agencijsku proviziju plaća`
- `Dozvole i potvrde` (100% None)
- `Tip kuće` (100% None)
- `Vrsta kuće (gradnje)` (100% None)
- `Površina okućnice` (100% None)
- `Pogled na more` (100% None)
- `Šifra objekta`
- `Namještenost i stanje` (79% None)
- `Netto površina` (79% None)
- `Kupaonica i WC - Broj kupaonica bez WC-a` (93% None)
- `Agencijska provizija` (96% None)

---

### NUMERIČKE TRANSFORMACIJE

**cijena** - isto kao kuće

**Stambena površina** - isto kao kuće

**Broj parkirnih mjesta**
- None → NaN
- "nema vlastito parkirno mjesto" → 0
- "7+" → 7
- Ostalo pretvoriti u int

**Godina izgradnje** - isto kao kuće (s flagom)

**Godina zadnje renovacije** - isto kao kuće

---

### MAPIRANJE KATEGORIJA → BROJ

**Broj etaža**
| Vrijednost | Nova vrijednost |
|------------|-----------------|
| Jednoetažni | 1 |
| Dvoetažni | 2 |
| Višeetažni | 3 |

**Broj soba**
| Vrijednost | Nova vrijednost |
|------------|-----------------|
| Garsonijera | 0.5 |
| 1-sobni | 1 |
| 2-sobni | 2 |
| 3-sobni | 3 |
| 4-sobni | 4 |
| 5+ sobni | 5 |

**Energetski razred** - isto kao kuće (s flagom)

**Kupaonica i WC - Broj WC-a** - isto kao kuće

**Kupaonica i WC - Broj kupaonica s WC-om** - isto kao kuće

**Ukupni broj katova**
- "Prizemlje" → 0
- "Visoko prizemlje" → 0.5
- "25+" → 25
- Ukloniti "." ako postoji
- None → NaN
- Dodati: `ima_ukupni_broj_katova` flag

**Kat**
- "Suteren" → -1
- "Prizemlje" → 0
- "Visoko prizemlje" → 0.5
- Brojevi (npr. "5.") → ukloniti ".", pretvoriti u int
- "Potkrovlje", "Visoko potkrovlje", "Penthouse" → NaN (rješavamo flagovima)
- None → NaN

Dodati flag stupce:
- `kat_suteren` (1 ako je Suteren, inače 0)
- `kat_prizemlje` (1 ako je Prizemlje ili Visoko prizemlje, inače 0)
- `kat_potkrovlje` (1 ako je Potkrovlje ili Visoko potkrovlje, inače 0)
- `kat_penthouse` (1 ako je Penthouse, inače 0)

---

### BINARNO MAPIRANJE (Da/None → 1/0)

| Stupac | None → | Da → |
|--------|--------|------|
| Grijanje - Dodatni izvor grijanja | 0 | 1 |
| Grijanje - Klima uređaj | 0 | 1 |
| Razgledavanje putem video poziva | 0 | 1 |

**Mogućnost zamjene** - isto kao kuće

---

### ONE-HOT ENCODING (multi-value stupci)

**Balkon/Lođa/Terasa** - isto kao kuće

**Dozvole** → `dozvola_`
- Jedinstvene vrijednosti: Vlasnički list, Građevinska dozvola, Uporabna dozvola

**Funkcionalnosti i ostale karakteristike** - isto kao kuće

**Grijanje - Alternativni izvori energije** - isto kao kuće (+ Kogeneracija)

**Grijanje - Sustav grijanja** - isto kao kuće (+ Zajednička kotlovnica)

**Orijentacija stana** → `orijentacija_`
- Jedinstvene vrijednosti: Sjever, Jug, Zapad, Istok

**Ostali objekti i površine** - isto kao kuće

**Podaci o objektu** - isto kao kuće

**Tip stana** → `tip_stana_`
- Jedinstvene vrijednosti: U stambenoj zgradi, U kući

**Vrsta parkinga** - isto kao kuće

---

### LOKACIJA - RAZDVAJANJE

Isto kao kuće → `zupanija`, `grad_opcina`, `naselje`

---

## SAŽETAK FINALNIH STUPACA

### Kuće (~25-30 stupaca nakon transformacije)
- Numerički: cijena, stambena_povrsina, povrsina_okucnice, broj_soba, broj_parkirnih_mjesta, broj_etaza, energetski_razred, godina_izgradnje, godina_renovacije, wc_broj, kupaonica_s_wc_broj
- Binarni: ima_godinu_izgradnje, ima_energetski_razred, ima_sustav_grijanja, grijanje_dodatni, grijanje_klima, pogled_more, video_poziv, mogucnost_zamjene
- One-hot: balkon_*, dozvola_*, funk_*, alt_energija_*, grijanje_*, objekt_*, podatak_*, tip_kuce_*, gradnja_*, parking_*
- Kategorički: zupanija, grad_opcina, naselje

### Stanovi (~30-35 stupaca nakon transformacije)
- Numerički: cijena, stambena_povrsina, broj_soba, broj_parkirnih_mjesta, broj_etaza, energetski_razred, godina_izgradnje, godina_renovacije, kat, ukupni_broj_katova, wc_broj, kupaonica_s_wc_broj
- Binarni: ima_godinu_izgradnje, ima_energetski_razred, ima_sustav_grijanja, ima_ukupni_broj_katova, kat_suteren, kat_prizemlje, kat_potkrovlje, kat_penthouse, grijanje_dodatni, grijanje_klima, video_poziv, mogucnost_zamjene
- One-hot: balkon_*, dozvola_*, funk_*, alt_energija_*, grijanje_*, orijentacija_*, objekt_*, podatak_*, tip_stana_*, parking_*
- Kategorički: zupanija, grad_opcina, naselje
