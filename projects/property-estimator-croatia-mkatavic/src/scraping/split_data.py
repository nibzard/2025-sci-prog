import pandas as pd

# Ucitaj podatke
df = pd.read_parquet('data/raw/properties_raw.parquet')

# Podijeli na stanove i kuce
stanovi = df[df['vrsta_nekretnine'] == 'stan']
kuce = df[df['vrsta_nekretnine'] == 'kuca']

# Spremi u zasebne datoteke
stanovi.to_parquet('data/raw/apartments.parquet', index=False)
kuce.to_parquet('data/raw/houses.parquet', index=False)

print(f"Stanovi: {len(stanovi)} zapisa -> data/raw/apartments.parquet")
print(f"Kuce: {len(kuce)} zapisa -> data/raw/houses.parquet")
