"""Generate EDA and form data for the web application."""
import pandas as pd
import numpy as np
import json
from pathlib import Path

# Paths
ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = Path(__file__).parent / "public" / "data"

# Load data
df_apt = pd.read_parquet(DATA_DIR / "apartments_clean.parquet")
df_house = pd.read_parquet(DATA_DIR / "houses_clean.parquet")

def build_location_hierarchy(df):
    """Build hierarchical location data: zupanija -> grad -> naselje"""
    hierarchy = {}

    for _, row in df[['zupanija', 'grad_opcina', 'naselje']].drop_duplicates().iterrows():
        zup = row['zupanija']
        grad = row['grad_opcina']
        naselje = row['naselje']

        if zup not in hierarchy:
            hierarchy[zup] = {}
        if grad not in hierarchy[zup]:
            hierarchy[zup][grad] = []
        if naselje and pd.notna(naselje) and naselje not in hierarchy[zup][grad]:
            hierarchy[zup][grad].append(naselje)

    # Sort everything
    sorted_hierarchy = {}
    for zup in sorted(hierarchy.keys()):
        sorted_hierarchy[zup] = {}
        for grad in sorted(hierarchy[zup].keys()):
            sorted_hierarchy[zup][grad] = sorted(hierarchy[zup][grad]) if hierarchy[zup][grad] else []

    return sorted_hierarchy

def generate_detailed_eda(df, property_type):
    """Generate detailed EDA statistics"""

    # Basic stats
    result = {
        'total_count': len(df),
        'avg_price': float(df['cijena'].mean()),
        'median_price': float(df['cijena'].median()),
        'min_price': float(df['cijena'].min()),
        'max_price': float(df['cijena'].max()),
        'std_price': float(df['cijena'].std()),
        'avg_area': float(df['stambena_povrsina'].mean()),
        'median_area': float(df['stambena_povrsina'].median()),
    }

    # Price per m2
    price_per_m2 = df['cijena'] / df['stambena_povrsina']
    result['avg_price_per_m2'] = float(price_per_m2.mean())
    result['median_price_per_m2'] = float(price_per_m2.median())

    # By zupanija
    by_zup = df.groupby('zupanija').agg({
        'cijena': ['mean', 'median', 'min', 'max', 'count', 'std'],
        'stambena_povrsina': ['mean', 'median']
    }).reset_index()
    by_zup.columns = ['zupanija', 'price_mean', 'price_median', 'price_min', 'price_max', 'count', 'price_std', 'area_mean', 'area_median']
    by_zup['price_per_m2'] = by_zup['price_mean'] / by_zup['area_mean']
    result['by_zupanija'] = by_zup.round(0).replace({np.nan: None}).to_dict('records')

    # By grad
    by_grad = df.groupby('grad_opcina').agg({
        'cijena': ['mean', 'median', 'min', 'max', 'count', 'std'],
        'stambena_povrsina': ['mean', 'median']
    }).reset_index()
    by_grad.columns = ['grad_opcina', 'price_mean', 'price_median', 'price_min', 'price_max', 'count', 'price_std', 'area_mean', 'area_median']
    by_grad['price_per_m2'] = by_grad['price_mean'] / by_grad['area_mean']
    result['by_grad'] = by_grad.round(0).replace({np.nan: None}).to_dict('records')

    # Price distribution (histogram bins)
    bins = list(range(0, 1000001, 50000)) + [float('inf')]
    df_copy = df.copy()
    df_copy['price_bin'] = pd.cut(df_copy['cijena'], bins=bins, right=False)
    price_dist = df_copy['price_bin'].value_counts().sort_index()
    result['price_distribution'] = [
        {'min': int(interval.left), 'max': int(min(interval.right, 1000000)), 'count': int(count)}
        for interval, count in price_dist.items() if count > 0
    ]

    # Area distribution
    area_bins = list(range(0, 301, 20)) + [float('inf')]
    df_copy['area_bin'] = pd.cut(df_copy['stambena_povrsina'], bins=area_bins, right=False)
    area_dist = df_copy['area_bin'].value_counts().sort_index()
    result['area_distribution'] = [
        {'min': int(interval.left), 'max': int(min(interval.right, 300)), 'count': int(count)}
        for interval, count in area_dist.items() if count > 0
    ]

    # Price per m2 distribution
    m2_bins = list(range(0, 10001, 500)) + [float('inf')]
    df_copy['price_per_m2'] = df_copy['cijena'] / df_copy['stambena_povrsina']
    df_copy['m2_bin'] = pd.cut(df_copy['price_per_m2'], bins=m2_bins, right=False)
    m2_dist = df_copy['m2_bin'].value_counts().sort_index()
    result['price_per_m2_distribution'] = [
        {'min': int(interval.left), 'max': int(min(interval.right, 10000)), 'count': int(count)}
        for interval, count in m2_dist.items() if count > 0
    ]

    # Year distribution
    year_data = df['godina_izgradnje'].dropna()
    if len(year_data) > 0:
        year_bins = list(range(1900, 2031, 10))
        df_year = df.dropna(subset=['godina_izgradnje']).copy()
        df_year['year_bin'] = pd.cut(df_year['godina_izgradnje'], bins=year_bins, right=False)
        year_dist = df_year['year_bin'].value_counts().sort_index()
        result['year_distribution'] = [
            {'decade': f"{int(interval.left)}s", 'min': int(interval.left), 'max': int(interval.right), 'count': int(count)}
            for interval, count in year_dist.items() if count > 0
        ]

    # Rooms distribution
    rooms_dist = df['broj_soba'].value_counts().sort_index()
    result['rooms_distribution'] = [
        {'rooms': float(k), 'count': int(v)} for k, v in rooms_dist.items()
    ]

    # Scatter data (all records for per-city distribution computation)
    scatter_cols = ['stambena_povrsina', 'cijena', 'zupanija', 'grad_opcina', 'broj_soba', 'godina_izgradnje']
    scatter_df = df[scatter_cols].copy()
    records = scatter_df.to_dict('records')
    # Replace NaN with None for valid JSON serialization
    for rec in records:
        for k, v in rec.items():
            if isinstance(v, float) and np.isnan(v):
                rec[k] = None
    result['scatter'] = records

    # Top 10 most expensive cities
    by_grad_df = pd.DataFrame(result['by_grad'])
    top_cities = by_grad_df.nlargest(10, 'price_mean')[['grad_opcina', 'price_mean', 'count']].to_dict('records')
    result['top_expensive_cities'] = top_cities

    # Top 10 cheapest cities (with at least 10 listings)
    affordable = by_grad_df[by_grad_df['count'] >= 10]
    if len(affordable) > 0:
        bottom_cities = affordable.nsmallest(10, 'price_mean')[['grad_opcina', 'price_mean', 'count']].to_dict('records')
        result['top_affordable_cities'] = bottom_cities
    else:
        result['top_affordable_cities'] = []

    return result

# Generate EDA for both types
print("Generating apartment EDA...")
apt_eda = generate_detailed_eda(df_apt, 'apartments')
print("Generating house EDA...")
house_eda = generate_detailed_eda(df_house, 'houses')

# City data with coordinates
city_coords = {
    'Zagreb': (45.815, 15.982), 'Split': (43.508, 16.440), 'Rijeka': (45.327, 14.442),
    'Osijek': (45.551, 18.694), 'Zadar': (44.119, 15.232), 'Pula': (44.867, 13.850),
    'Slavonski Brod': (45.160, 18.016), 'Karlovac': (45.487, 15.548),
    'Sisak': (45.466, 16.378), 'Velika Gorica': (45.714, 16.075),
}

apt_by_grad = {g['grad_opcina']: g for g in apt_eda['by_grad']}
house_by_grad = {g['grad_opcina']: g for g in house_eda['by_grad']}
apt_by_zup = {z['zupanija']: z for z in apt_eda['by_zupanija']}
house_by_zup = {z['zupanija']: z for z in house_eda['by_zupanija']}

cities = []
for city, (lat, lng) in city_coords.items():
    apt_data = apt_by_grad.get(city, apt_by_zup.get('Grad ' + city if city == 'Zagreb' else city))
    house_data = house_by_grad.get(city, house_by_zup.get('Grad ' + city if city == 'Zagreb' else city))

    apt_count = int(apt_data['count']) if apt_data else 0
    apt_avg = int(apt_data['price_mean']) if apt_data else 0
    house_count = int(house_data['count']) if house_data else 0
    house_avg = int(house_data['price_mean']) if house_data else 0

    if apt_count > 0 or house_count > 0:
        cities.append({
            'name': city, 'lat': lat, 'lng': lng,
            'apt_count': apt_count, 'apt_avg_price': apt_avg,
            'house_count': house_count, 'house_avg_price': house_avg
        })

cities.sort(key=lambda x: x['apt_count'] + x['house_count'], reverse=True)

# Save EDA data
eda_data = {
    'apartments': apt_eda,
    'houses': house_eda,
    'cities': cities
}

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_DIR / 'eda.json', 'w', encoding='utf-8') as f:
    json.dump(eda_data, f, ensure_ascii=False)

print(f"Generated detailed EDA data")
print(f"  Apartments: {apt_eda['total_count']} listings")
print(f"  Houses: {house_eda['total_count']} listings")
print(f"  Cities: {len(cities)}")

# Generate form data with hierarchies
print("\nGenerating form data with location hierarchies...")
apt_hierarchy = build_location_hierarchy(df_apt)
house_hierarchy = build_location_hierarchy(df_house)

# Load existing form.json if it exists
form_path = OUTPUT_DIR / 'form.json'
if form_path.exists():
    with open(form_path, 'r', encoding='utf-8') as f:
        form_data = json.load(f)
else:
    form_data = {'apartments': {}, 'houses': {}}

# Add location hierarchies
form_data['apartments']['location_hierarchy'] = apt_hierarchy
form_data['houses']['location_hierarchy'] = house_hierarchy

# Also add flat lists for backwards compatibility
form_data['apartments']['zupanije'] = sorted(apt_hierarchy.keys())
form_data['apartments']['gradovi'] = sorted(set(g for zup in apt_hierarchy.values() for g in zup.keys()))
form_data['apartments']['naselja'] = sorted(set(n for zup in apt_hierarchy.values() for grads in zup.values() for n in grads))

form_data['houses']['zupanije'] = sorted(house_hierarchy.keys())
form_data['houses']['gradovi'] = sorted(set(g for zup in house_hierarchy.values() for g in zup.keys()))
form_data['houses']['naselja'] = sorted(set(n for zup in house_hierarchy.values() for grads in zup.values() for n in grads))

with open(form_path, 'w', encoding='utf-8') as f:
    json.dump(form_data, f, ensure_ascii=False, indent=2)

print(f"  Apartments: {len(apt_hierarchy)} županije")
print(f"  Houses: {len(house_hierarchy)} županije")
print("\nDone!")
