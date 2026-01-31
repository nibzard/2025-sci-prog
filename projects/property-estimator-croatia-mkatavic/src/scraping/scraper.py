#!/usr/bin/env python3
"""
Scraper za prikupljanje podataka o nekretninama.
"""

import hashlib
import json
import os
import random
import re
import time
from io import BytesIO
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from PIL import Image
from steel import Steel

# Ucitaj environment varijable
load_dotenv()

# Konfiguracija
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent  # src/scraping -> src -> project root
DATA_DIR = PROJECT_ROOT / "data"
IMAGES_DIR = DATA_DIR / "images"
RAW_DATA_DIR = DATA_DIR / "raw"

IMAGES_DIR.mkdir(parents=True, exist_ok=True)
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = (512, 512)
IMAGE_QUALITY = 85
MAX_IMAGES_PER_LISTING = 5

# Domena za slike
# IMAGE_HOST_DOMAIN = ""

# Steel konfiguracija
STEEL_API_KEY = os.getenv("STEEL_API_KEY")


def create_session() -> requests.Session:
    """Stvara session s realisticnim headerima."""
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
        'Accept-Language': 'hr-HR,hr;q=0.9,en-US;q=0.8,en;q=0.7',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'none',
        'Sec-Fetch-User': '?1',
        'Cache-Control': 'max-age=0',
    })
    return session


def get_steel_client():
    """Vraca Steel klijent."""
    return Steel(steel_api_key=STEEL_API_KEY)


def fetch_html_with_steel(url: str) -> str | None:
    """Dohvaca HTML sadrzaj s URL-a koristeci Steel API."""
    for attempt in range(3):
        try:
            client = get_steel_client()
            response = client.scrape(url=url, delay=3000)

            if response and response.content and response.content.html:
                return response.content.html
            return None

        except Exception as e:
            error_str = str(e).lower()

            if "429" in str(e) or "quota" in error_str or "limit" in error_str or "unauthorized" in error_str or "402" in str(e):
                print(f"  Steel API limit iscrpljen")
                return None

            if attempt < 2:
                print(f"  Steel greska (pokusaj {attempt + 1}): {e}")
                time.sleep(2)
            else:
                print(f"  Steel greska: {e}")
                return None

    return None


def generate_listing_id(url: str) -> str:
    """Generira jedinstveni ID za oglas baziran na URL-u."""
    return hashlib.md5(url.encode()).hexdigest()[:12]


def download_and_save_image(
    session: requests.Session,
    url: str,
    listing_id: str,
    image_index: int
) -> str | None:
    """
    Preuzima sliku, resizea (zadrzava aspect ratio) i sprema kao WebP.
    Vraca putanju do spremljene slike ili None ako nije uspjelo.
    """
    try:
        response = session.get(url, timeout=15)
        response.raise_for_status()

        img = Image.open(BytesIO(response.content))
        img = img.convert("RGB")

        original_size = img.size

        # Koristi thumbnail - zadrzava aspect ratio, smanjuje samo ako je vece
        # Slika ce stati unutar IMAGE_SIZE box-a bez distorzije
        img.thumbnail(IMAGE_SIZE, Image.Resampling.LANCZOS)

        # Spremi kao WebP
        filename = f"{listing_id}_{image_index}.webp"
        filepath = IMAGES_DIR / filename
        img.save(filepath, "WEBP", quality=IMAGE_QUALITY)

        # Debug: prikazi ako je slika bila manja od ocekivanog
        if original_size[0] < IMAGE_SIZE[0] and original_size[1] < IMAGE_SIZE[1]:
            print(f"    [!] Mala izvorna slika: {original_size}")

        return str(filepath)

    except Exception:
        return None


def select_spaced_items(items: list, count: int) -> list:
    """
    Odabire 'count' elemenata ravnomjerno rasporedenih (space-between).
    Uvijek ukljucuje prvi i zadnji element.
    """
    n = len(items)
    if n <= count:
        return items

    # Indeksi: prvi, zadnji, i (count-2) ravnomjerno rasporedenih izmedu
    indices = [0]  # prvi

    # Izracunaj srednje indekse (count-2 komada izmedu prvog i zadnjeg)
    if count > 2:
        step = (n - 1) / (count - 1)
        for i in range(1, count - 1):
            indices.append(round(i * step))

    indices.append(n - 1)  # zadnji

    return [items[i] for i in indices]


def extract_image_urls(soup: BeautifulSoup, limit: int = MAX_IMAGES_PER_LISTING) -> list[str]:
    """Ekstrahira URL-ove slika iz oglasa (puna rezolucija, space-between odabir)."""
    all_image_urls = []

    # Trazi slike u galeriji - prioritet data-src (lazy load), pa src
    gallery_images = soup.select('.ClassifiedDetailGallery-image img, .ClassifiedDetailGallery img')

    for img in gallery_images:
        # Preferiraj data-src jer sadrzi punu rezoluciju
        src = img.get('data-src') or img.get('src')

        if not src or IMAGE_HOST_DOMAIN not in src:
            continue

        # Preskoci male thumbnaile (80x60, 45x60, itd.)
        if '/image-80x60/' in src or '/image-45x60/' in src:
            continue

        # Osiguraj da koristimo veliku verziju (920x690)
        # Zamijeni bilo koji image-XXX format s image-w920x690
        src = re.sub(r'/image-\d+x\d+/', '/image-w920x690/', src)
        src = re.sub(r'/image-small/', '/image-w920x690/', src)
        src = re.sub(r'/image-medium/', '/image-w920x690/', src)
        src = re.sub(r'/image-big/', '/image-w920x690/', src)

        if src not in all_image_urls:
            all_image_urls.append(src)

    # Odaberi ravnomjerno rasporedene slike (space-between)
    return select_spaced_items(all_image_urls, limit)


def extract_description(soup: BeautifulSoup) -> str | None:
    """Ekstrahira opis oglasa."""
    desc_elem = soup.select_one('.ClassifiedDetailDescription-text')
    if desc_elem:
        return desc_elem.get_text(strip=True)
    return None


def extract_price(soup: BeautifulSoup) -> str | None:
    """Ekstrahira cijenu iz oglasa."""
    price_elem = soup.select_one('.ClassifiedDetailSummary-priceDomestic')
    if price_elem:
        return price_elem.get_text(strip=True)
    return None


def extract_basic_info(soup: BeautifulSoup) -> dict:
    """Ekstrahira osnovne informacije (Lokacija, Tip stana, Broj soba, itd.)."""
    basic_info = {}

    basic_details = soup.select('.ClassifiedDetailBasicDetails-list dt, .ClassifiedDetailBasicDetails-list dd')

    current_key = None
    for elem in basic_details:
        if elem.name == 'dt':
            text_container = elem.select_one('.ClassifiedDetailBasicDetails-textWrapContainer')
            if text_container:
                current_key = text_container.get_text(strip=True)
        elif elem.name == 'dd' and current_key:
            text_container = elem.select_one('.ClassifiedDetailBasicDetails-textWrapContainer')
            if text_container:
                value = text_container.get_text(strip=True)
                basic_info[current_key] = value
            current_key = None

    return basic_info


def extract_additional_info(soup: BeautifulSoup) -> dict:
    """Ekstrahira dodatne informacije (Grijanje, Dozvole, itd.)."""
    additional_info = {}

    property_groups = soup.select('.ClassifiedDetailPropertyGroups-group')

    for group in property_groups:
        group_title_elem = group.select_one('.ClassifiedDetailPropertyGroups-groupTitle')
        if not group_title_elem:
            continue

        group_title = group_title_elem.get_text(strip=True)
        items = group.select('.ClassifiedDetailPropertyGroups-groupListItem')

        values = []
        for item in items:
            item_text = item.get_text(strip=True)

            if ':' in item_text:
                key, value = item_text.split(':', 1)
                attr_name = f"{group_title} - {key.strip()}"
                additional_info[attr_name] = value.strip()
            else:
                values.append(item_text)

        if values:
            additional_info[group_title] = ', '.join(values)

    return additional_info


def scrape_listing(
    html_content: str,
    url: str,
    grad: str,
    vrsta: str,
    session: requests.Session
) -> dict | None:
    """
    Parsira HTML i sprema sve podatke ukljucujuci slike.
    Vraca dict s podacima ili None ako scraping nije uspio.
    """
    soup = BeautifulSoup(html_content, 'html.parser')

    listing_id = generate_listing_id(url)

    data = {
        'listing_id': listing_id,
        'url': url,
        'grad': grad,
        'vrsta_nekretnine': vrsta,
        'scraped_at': pd.Timestamp.now().isoformat(),
    }

    # Cijena
    price = extract_price(soup)
    if price:
        data['cijena'] = price

    # Osnovne informacije
    basic_info = extract_basic_info(soup)
    data.update(basic_info)

    # Dodatne informacije
    additional_info = extract_additional_info(soup)
    data.update(additional_info)

    # Opis (raw text za kasniju batch analizu)
    description = extract_description(soup)
    if description:
        data['opis'] = description

    # Slike - preuzmi i spremi lokalno
    image_urls = extract_image_urls(soup)
    saved_images = []

    for idx, img_url in enumerate(image_urls):
        saved_path = download_and_save_image(session, img_url, listing_id, idx)
        if saved_path:
            saved_images.append(saved_path)

    data['image_paths'] = json.dumps(saved_images)  # JSON string za parquet
    data['image_count'] = len(saved_images)

    return data


def load_properties_from_json(json_path: str, limit: int | None = None) -> list[dict]:
    """Ucitava listu nekretnina iz JSON datoteke."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    properties = []

    for city, types in data.get('cities', {}).items():
        for url in types.get('apartments', []):
            properties.append({
                'url': url,
                'grad': city,
                'vrsta_nekretnine': 'stan'
            })

        for url in types.get('houses', []):
            properties.append({
                'url': url,
                'grad': city,
                'vrsta_nekretnine': 'kuca'
            })

    if limit:
        properties = properties[:limit]

    return properties


def load_existing_data(parquet_path: str) -> set[str]:
    """Ucitava postojece URL-ove da izbjegnemo duplikate."""
    if Path(parquet_path).exists():
        df = pd.read_parquet(parquet_path)
        return set(df['url'].tolist())
    return set()


def main():
    """Glavna funkcija - scraping bez AI analize."""
    import argparse

    parser = argparse.ArgumentParser(description="Scraper za nekretnine")
    parser.add_argument("--limit", type=int, default=None, help="Maksimalan broj oglasa za scrapanje")
    args = parser.parse_args()

    json_path = SCRIPT_DIR / 'properties.json'
    output_file = RAW_DATA_DIR / 'properties_raw.parquet'

    if not json_path.exists():
        print(f"Greska: Datoteka '{json_path}' ne postoji.")
        return

    # Ucitaj nekretnine
    properties = load_properties_from_json(json_path, limit=args.limit)

    if not properties:
        print("Nema nekretnina za obradu.")
        return

    # Ucitaj vec scrapane URL-ove
    existing_urls = load_existing_data(output_file)
    properties = [p for p in properties if p['url'] not in existing_urls]

    total_in_json = len(load_properties_from_json(json_path))
    print(f"Ukupno oglasa u bazi: {total_in_json}")
    print(f"Vec scrapanih: {len(existing_urls)}")
    print(f"Preostalo za scraping: {len(properties)}")
    print(f"Slike ce biti spremljene u: {IMAGES_DIR}")
    print(f"Rezolucija slika: {IMAGE_SIZE[0]}x{IMAGE_SIZE[1]} WebP")
    print("-" * 50)

    session = create_session()
    all_listings = []

    for i, prop in enumerate(properties, 1):
        url = prop['url']
        grad = prop['grad']
        vrsta = prop['vrsta_nekretnine']

        print(f"[{i}/{len(properties)}] {url}")

        html_content = fetch_html_with_steel(url)
        if not html_content:
            continue

        try:
            listing_data = scrape_listing(html_content, url, grad, vrsta, session)
            if listing_data:
                all_listings.append(listing_data)
                img_count = listing_data.get('image_count', 0)
                print(f"  OK - {len(listing_data)} atributa, {img_count} slika")
        except Exception as e:
            print(f"  Greska: {e}")

        # Pauza izmedu zahtjeva
        time.sleep(random.uniform(2, 4))

        # Spremi svakih 10 oglasa (incremental save)
        if len(all_listings) % 10 == 0 and all_listings:
            save_data(all_listings, output_file)

    # Finalno spremanje
    if all_listings:
        save_data(all_listings, output_file)


def save_data(new_listings: list[dict], output_file: str):
    """Sprema podatke u parquet (append ako postoji)."""
    new_df = pd.DataFrame(new_listings)

    if Path(output_file).exists():
        existing_df = pd.read_parquet(output_file)
        df = pd.concat([existing_df, new_df], ignore_index=True)
        df = df.drop_duplicates(subset=['url'], keep='last')
    else:
        df = new_df

    # Sortiraj stupce
    priority_cols = ['listing_id', 'url', 'grad', 'vrsta_nekretnine', 'cijena',
                     'opis', 'image_paths', 'image_count', 'scraped_at']
    other_cols = sorted([c for c in df.columns if c not in priority_cols])
    df = df[[c for c in priority_cols if c in df.columns] + other_cols]

    df.to_parquet(output_file, index=False)
    print(f"  -> Spremljeno {len(df)} oglasa u {output_file}")


if __name__ == '__main__':
    main()
