"""
Script za čišćenje podataka za Croatian Property Price Estimator.
Implementira plan definiran u data_cleaning_plan.md
"""

import pandas as pd
import numpy as np


def parse_price(value):
    """Pretvara cijenu iz formata '330.000 €' u float."""
    if pd.isna(value):
        return np.nan
    value = str(value).replace(' €', '').replace('.', '').replace(',', '.')
    try:
        return float(value)
    except ValueError:
        return np.nan


def parse_area(value):
    """Pretvara površinu iz formata '139,27 m²' u float."""
    if pd.isna(value):
        return np.nan
    value = str(value).replace(' m²', '').replace('.', '').replace(',', '.')
    try:
        return float(value)
    except ValueError:
        return np.nan


def parse_year(value):
    """Pretvara godinu iz formata '2025.' u int."""
    if pd.isna(value):
        return np.nan
    value = str(value).replace('.', '')
    try:
        return int(value)
    except ValueError:
        return np.nan


def parse_bathroom_count(value):
    """Pretvara broj kupaonica/WC-a, '5+' -> 5."""
    if pd.isna(value):
        return np.nan
    if value == '5+':
        return 5
    try:
        return int(value)
    except ValueError:
        return np.nan


def binary_mapping(value, positive_values=['Da']):
    """Mapira Da/None u 1/0."""
    if pd.isna(value):
        return 0
    return 1 if value in positive_values else 0


def multi_hot_encode(df, column, prefix, separator=', '):
    """
    Radi one-hot encoding za stupce s više vrijednosti.
    Vraća DataFrame s novim stupcima.
    """
    # Pronađi sve jedinstvene vrijednosti
    all_values = set()
    for val in df[column].dropna():
        if val and val != 'None' and val != 'Nema ništa navedeno':
            for item in str(val).split(separator):
                item = item.strip()
                if item:
                    all_values.add(item)

    # Kreiraj stupce za svaku vrijednost
    new_columns = {}
    for unique_val in sorted(all_values):
        col_name = f"{prefix}{unique_val.lower().replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '').replace('-', '_')}"
        new_columns[col_name] = df[column].apply(
            lambda x: 1 if pd.notna(x) and unique_val in str(x) else 0
        )

    return pd.DataFrame(new_columns)


def split_location(df):
    """Razdvaja Lokacija stupac u zupanija, grad_opcina, naselje."""
    def extract_parts(value):
        if pd.isna(value):
            return pd.Series([np.nan, np.nan, np.nan])
        parts = str(value).split(', ')
        if len(parts) >= 3:
            return pd.Series([parts[0].strip(), parts[1].strip(), parts[2].strip()])
        elif len(parts) == 2:
            return pd.Series([parts[0].strip(), parts[1].strip(), np.nan])
        elif len(parts) == 1:
            return pd.Series([parts[0].strip(), np.nan, np.nan])
        return pd.Series([np.nan, np.nan, np.nan])

    location_parts = df['Lokacija'].apply(extract_parts)
    location_parts.columns = ['zupanija', 'grad_opcina', 'naselje']
    return location_parts


def clean_houses(df):
    """Cisti dataset kuca prema planu."""
    print("Ciscenje dataseta kuca...")
    print(f"Pocetni broj redaka: {len(df)}")
    print(f"Pocetni broj stupaca: {len(df.columns)}")

    result = pd.DataFrame()

    # ===== NUMERIČKE TRANSFORMACIJE =====

    # Cijena
    result['cijena'] = df['cijena'].apply(parse_price)

    # Stambena površina
    result['stambena_povrsina'] = df['Stambena površina'].apply(parse_area)

    # Površina okućnice
    result['povrsina_okucnice'] = df['Površina okućnice'].apply(parse_area)

    # Broj soba (već numerički)
    result['broj_soba'] = pd.to_numeric(df['Broj soba'], errors='coerce')

    # Broj parkirnih mjesta
    result['broj_parkirnih_mjesta'] = pd.to_numeric(df['Broj parkirnih mjesta'], errors='coerce')

    # Godina izgradnje
    result['godina_izgradnje'] = df['Godina izgradnje'].apply(parse_year)
    result['ima_godinu_izgradnje'] = result['godina_izgradnje'].notna().astype(int)

    # Godina zadnje renovacije
    result['godina_renovacije'] = df['Godina zadnje renovacije'].apply(parse_year)
    # Gdje nema renovacije, a ima izgradnju -> kopiraj godinu izgradnje
    mask = result['godina_renovacije'].isna() & result['godina_izgradnje'].notna()
    result.loc[mask, 'godina_renovacije'] = result.loc[mask, 'godina_izgradnje']

    # ===== MAPIRANJE KATEGORIJA → BROJ =====

    # Broj etaža
    etaze_map = {
        'Prizemnica': 1,
        'Visoka prizemnica': 1.5,
        'Katnica': 2,
        'Dvokatnica': 3,
        'Višekatnica': 4
    }
    result['broj_etaza'] = df['Broj etaža'].map(etaze_map)

    # Energetski razred
    energetski_map = {
        'A+': 5, 'A': 4, 'B': 3, 'C': 2, 'D': 1, 'E': 0, 'F': -1, 'G': -2
    }
    result['energetski_razred'] = df['Energetski razred'].map(energetski_map)
    result['ima_energetski_razred'] = df['Energetski razred'].notna().astype(int)

    # Kupaonica i WC
    result['wc_broj'] = df['Kupaonica i WC - Broj WC-a'].apply(parse_bathroom_count)
    result['kupaonica_s_wc_broj'] = df['Kupaonica i WC - Broj kupaonica s WC-om'].apply(parse_bathroom_count)

    # ===== BINARNO MAPIRANJE =====

    result['grijanje_dodatni'] = df['Grijanje - Dodatni izvor grijanja'].apply(binary_mapping)
    result['grijanje_klima'] = df['Grijanje - Klima uređaj'].apply(binary_mapping)
    result['pogled_more'] = df['Pogled na more'].apply(binary_mapping)
    result['video_poziv'] = df['Razgledavanje putem video poziva'].apply(binary_mapping)

    # Mogućnost zamjene
    result['mogucnost_zamjene'] = df['Mogućnost zamjene'].apply(
        lambda x: 1 if x == 'Moguća zamjena za drugu nekretninu' else 0
    )

    # ===== ONE-HOT ENCODING =====

    # Balkon/Lođa/Terasa
    balkon_encoded = multi_hot_encode(df, 'Balkon/Lođa/Terasa', 'balkon_')
    result = pd.concat([result, balkon_encoded], axis=1)

    # Dozvole i potvrde
    dozvole_encoded = multi_hot_encode(df, 'Dozvole i potvrde', 'dozvola_')
    result = pd.concat([result, dozvole_encoded], axis=1)

    # Funkcionalnosti i ostale karakteristike
    funk_encoded = multi_hot_encode(df, 'Funkcionalnosti i ostale karakteristike', 'funk_')
    result = pd.concat([result, funk_encoded], axis=1)

    # Grijanje - Alternativni izvori energije
    alt_energija_encoded = multi_hot_encode(df, 'Grijanje - Alternativni izvori energije', 'alt_energija_')
    result = pd.concat([result, alt_energija_encoded], axis=1)

    # Grijanje - Sustav grijanja
    grijanje_encoded = multi_hot_encode(df, 'Grijanje - Sustav grijanja', 'grijanje_sustav_')
    result = pd.concat([result, grijanje_encoded], axis=1)
    result['ima_sustav_grijanja'] = df['Grijanje - Sustav grijanja'].notna().astype(int)

    # Ostali objekti i površine
    objekt_encoded = multi_hot_encode(df, 'Ostali objekti i površine', 'objekt_')
    result = pd.concat([result, objekt_encoded], axis=1)

    # Podaci o objektu
    podatak_encoded = multi_hot_encode(df, 'Podaci o objektu', 'podatak_')
    result = pd.concat([result, podatak_encoded], axis=1)

    # Tip kuće
    tip_kuce_encoded = multi_hot_encode(df, 'Tip kuće', 'tip_kuce_')
    result = pd.concat([result, tip_kuce_encoded], axis=1)

    # Vrsta kuće (gradnje)
    gradnja_encoded = multi_hot_encode(df, 'Vrsta kuće (gradnje)', 'gradnja_')
    result = pd.concat([result, gradnja_encoded], axis=1)

    # Vrsta parkinga
    parking_encoded = multi_hot_encode(df, 'Vrsta parkinga', 'parking_')
    result = pd.concat([result, parking_encoded], axis=1)

    # ===== LOKACIJA =====
    location_parts = split_location(df)
    result = pd.concat([result, location_parts], axis=1)

    print(f"Zavrsni broj stupaca: {len(result.columns)}")
    return result


def clean_apartments(df):
    """Cisti dataset stanova prema planu."""
    print("Ciscenje dataseta stanova...")
    print(f"Pocetni broj redaka: {len(df)}")
    print(f"Pocetni broj stupaca: {len(df.columns)}")

    result = pd.DataFrame()

    # ===== NUMERIČKE TRANSFORMACIJE =====

    # Cijena
    result['cijena'] = df['cijena'].apply(parse_price)

    # Stambena površina
    result['stambena_povrsina'] = df['Stambena površina'].apply(parse_area)

    # Broj parkirnih mjesta
    def parse_parking_apartments(value):
        if pd.isna(value):
            return np.nan
        if value == 'nema vlastito parkirno mjesto':
            return 0
        if value == '7+':
            return 7
        try:
            return int(value)
        except ValueError:
            return np.nan

    result['broj_parkirnih_mjesta'] = df['Broj parkirnih mjesta'].apply(parse_parking_apartments)

    # Godina izgradnje
    result['godina_izgradnje'] = df['Godina izgradnje'].apply(parse_year)
    result['ima_godinu_izgradnje'] = result['godina_izgradnje'].notna().astype(int)

    # Godina zadnje renovacije
    result['godina_renovacije'] = df['Godina zadnje renovacije'].apply(parse_year)
    mask = result['godina_renovacije'].isna() & result['godina_izgradnje'].notna()
    result.loc[mask, 'godina_renovacije'] = result.loc[mask, 'godina_izgradnje']

    # ===== MAPIRANJE KATEGORIJA → BROJ =====

    # Broj etaža
    etaze_map = {
        'Jednoetažni': 1,
        'Dvoetažni': 2,
        'Višeetažni': 3
    }
    result['broj_etaza'] = df['Broj etaža'].map(etaze_map)

    # Broj soba
    sobe_map = {
        'Garsonijera': 0.5,
        '1-sobni': 1,
        '2-sobni': 2,
        '3-sobni': 3,
        '4-sobni': 4,
        '5+ sobni': 5
    }
    result['broj_soba'] = df['Broj soba'].map(sobe_map)

    # Energetski razred
    energetski_map = {
        'A+': 5, 'A': 4, 'B': 3, 'C': 2, 'D': 1, 'E': 0, 'F': -1, 'G': -2
    }
    result['energetski_razred'] = df['Energetski razred'].map(energetski_map)
    result['ima_energetski_razred'] = df['Energetski razred'].notna().astype(int)

    # Kupaonica i WC
    result['wc_broj'] = df['Kupaonica i WC - Broj WC-a'].apply(parse_bathroom_count)
    result['kupaonica_s_wc_broj'] = df['Kupaonica i WC - Broj kupaonica s WC-om'].apply(parse_bathroom_count)

    # Ukupni broj katova
    def parse_ukupni_katovi(value):
        if pd.isna(value):
            return np.nan
        if value == 'Prizemlje':
            return 0
        if value == 'Visoko prizemlje':
            return 0.5
        if value == '25+':
            return 25
        try:
            return int(str(value).replace('.', ''))
        except ValueError:
            return np.nan

    result['ukupni_broj_katova'] = df['Ukupni broj katova'].apply(parse_ukupni_katovi)
    result['ima_ukupni_broj_katova'] = df['Ukupni broj katova'].notna().astype(int)

    # Kat
    def parse_kat(value):
        if pd.isna(value):
            return np.nan
        if value == 'Suteren':
            return -1
        if value == 'Prizemlje':
            return 0
        if value == 'Visoko prizemlje':
            return 0.5
        if value in ['Potkrovlje', 'Visoko potkrovlje', 'Penthouse']:
            return np.nan  # Rješavamo flagovima
        try:
            return int(str(value).replace('.', ''))
        except ValueError:
            return np.nan

    result['kat'] = df['Kat'].apply(parse_kat)

    # Kat flagovi
    result['kat_suteren'] = (df['Kat'] == 'Suteren').astype(int)
    result['kat_prizemlje'] = df['Kat'].isin(['Prizemlje', 'Visoko prizemlje']).astype(int)
    result['kat_potkrovlje'] = df['Kat'].isin(['Potkrovlje', 'Visoko potkrovlje']).astype(int)
    result['kat_penthouse'] = (df['Kat'] == 'Penthouse').astype(int)

    # ===== BINARNO MAPIRANJE =====

    result['grijanje_dodatni'] = df['Grijanje - Dodatni izvor grijanja'].apply(binary_mapping)
    result['grijanje_klima'] = df['Grijanje - Klima uređaj'].apply(binary_mapping)
    result['video_poziv'] = df['Razgledavanje putem video poziva'].apply(binary_mapping)

    # Mogućnost zamjene
    result['mogucnost_zamjene'] = df['Mogućnost zamjene'].apply(
        lambda x: 1 if x == 'Moguća zamjena za drugu nekretninu' else 0
    )

    # ===== ONE-HOT ENCODING =====

    # Balkon/Lođa/Terasa
    balkon_encoded = multi_hot_encode(df, 'Balkon/Lođa/Terasa', 'balkon_')
    result = pd.concat([result, balkon_encoded], axis=1)

    # Dozvole
    dozvole_encoded = multi_hot_encode(df, 'Dozvole', 'dozvola_')
    result = pd.concat([result, dozvole_encoded], axis=1)

    # Funkcionalnosti i ostale karakteristike
    funk_encoded = multi_hot_encode(df, 'Funkcionalnosti i ostale karakteristike', 'funk_')
    result = pd.concat([result, funk_encoded], axis=1)

    # Grijanje - Alternativni izvori energije
    alt_energija_encoded = multi_hot_encode(df, 'Grijanje - Alternativni izvori energije', 'alt_energija_')
    result = pd.concat([result, alt_energija_encoded], axis=1)

    # Grijanje - Sustav grijanja
    grijanje_encoded = multi_hot_encode(df, 'Grijanje - Sustav grijanja', 'grijanje_sustav_')
    result = pd.concat([result, grijanje_encoded], axis=1)
    result['ima_sustav_grijanja'] = df['Grijanje - Sustav grijanja'].notna().astype(int)

    # Orijentacija stana
    orijentacija_encoded = multi_hot_encode(df, 'Orijentacija stana', 'orijentacija_')
    result = pd.concat([result, orijentacija_encoded], axis=1)

    # Ostali objekti i površine
    objekt_encoded = multi_hot_encode(df, 'Ostali objekti i površine', 'objekt_')
    result = pd.concat([result, objekt_encoded], axis=1)

    # Podaci o objektu
    podatak_encoded = multi_hot_encode(df, 'Podaci o objektu', 'podatak_')
    result = pd.concat([result, podatak_encoded], axis=1)

    # Tip stana
    tip_stana_encoded = multi_hot_encode(df, 'Tip stana', 'tip_stana_')
    result = pd.concat([result, tip_stana_encoded], axis=1)

    # Vrsta parkinga
    parking_encoded = multi_hot_encode(df, 'Vrsta parkinga', 'parking_')
    result = pd.concat([result, parking_encoded], axis=1)

    # ===== LOKACIJA =====
    location_parts = split_location(df)
    result = pd.concat([result, location_parts], axis=1)

    print(f"Zavrsni broj stupaca: {len(result.columns)}")
    return result


def main():
    # Ucitaj podatke
    print("Ucitavanje podataka...")
    df_houses = pd.read_parquet('data/raw/houses.parquet')
    df_apartments = pd.read_parquet('data/raw/apartments.parquet')

    # Zamijeni string 'None' s pravim NaN
    df_houses = df_houses.replace('None', np.nan)
    df_apartments = df_apartments.replace('None', np.nan)

    # Ocisti podatke
    houses_clean = clean_houses(df_houses)
    apartments_clean = clean_apartments(df_apartments)

    # Ukloni retke bez cijene (target varijabla)
    houses_before = len(houses_clean)
    apartments_before = len(apartments_clean)

    houses_clean = houses_clean.dropna(subset=['cijena'])
    apartments_clean = apartments_clean.dropna(subset=['cijena'])

    houses_removed = houses_before - len(houses_clean)
    apartments_removed = apartments_before - len(apartments_clean)

    if houses_removed > 0:
        print(f"Uklonjeno {houses_removed} kuca bez cijene")
    if apartments_removed > 0:
        print(f"Uklonjeno {apartments_removed} stanova bez cijene")

    # Spremi ociscene podatke
    print("\nSpremanje ociscenih podataka...")
    houses_clean.to_parquet('data/processed/houses_clean.parquet', index=False)
    apartments_clean.to_parquet('data/processed/apartments_clean.parquet', index=False)

    # Ispisi statistike
    print("\n" + "="*50)
    print("SAZETAK")
    print("="*50)
    print(f"\nKuce:")
    print(f"  - Redaka: {len(houses_clean)}")
    print(f"  - Stupaca: {len(houses_clean.columns)}")

    print(f"\nStanovi:")
    print(f"  - Redaka: {len(apartments_clean)}")
    print(f"  - Stupaca: {len(apartments_clean.columns)}")

    # Provjeri missing values
    print("\n" + "="*50)
    print("MISSING VALUES - KUCE")
    print("="*50)
    missing_houses = houses_clean.isnull().sum()
    missing_houses = missing_houses[missing_houses > 0].sort_values(ascending=False)
    for col, count in missing_houses.items():
        pct = count / len(houses_clean) * 100
        print(f"  {col}: {count} ({pct:.1f}%)")

    print("\n" + "="*50)
    print("MISSING VALUES - STANOVI")
    print("="*50)
    missing_apartments = apartments_clean.isnull().sum()
    missing_apartments = missing_apartments[missing_apartments > 0].sort_values(ascending=False)
    for col, count in missing_apartments.items():
        pct = count / len(apartments_clean) * 100
        print(f"  {col}: {count} ({pct:.1f}%)")

    print("\nGotovo!")


if __name__ == '__main__':
    main()
