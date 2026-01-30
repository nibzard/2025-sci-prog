import marimo

__generated_with = "0.19.2"
app = marimo.App()


@app.cell
def _():
    import pandas as pd
    import requests
    import marimo as mo
    import matplotlib.pyplot as plt
    import seaborn as sns
    from io import StringIO
    import os
    import random
    import numpy as np
    return np, os, pd, plt, random, requests, sns


@app.cell
def _(requests):
    # 02.1 Extract features from dataset
    def get_zenodo_file_links():
        zenodo_api_url = "https://zenodo.org/api/records/3541657"
        headers = {"User-Agent": "python-requests/1.0"}

        response = requests.get(zenodo_api_url, headers=headers)
        print("Status code:", response.status_code)
        print("Content-Type:", response.headers.get("Content-Type"))

        record_data = response.json()

        file_links = {file["key"]: file["links"]["self"] for file in record_data["files"]}

        print(file_links)

    get_zenodo_file_links()
    return


@app.cell
def _(os, requests):
    def download_user_files():
        output_dir = "data/user_raw"
        os.makedirs(output_dir, exist_ok=True)

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0 Safari/537.36"
        }

        file_links = {
            "gender.txt": "https://zenodo.org/api/records/3541657/files/gender.txt/content",
            "age.txt": "https://zenodo.org/api/records/3541657/files/age.txt/content",
            "profession.txt": "https://zenodo.org/api/records/3541657/files/profession.txt/content",
            "hobby.txt": "https://zenodo.org/api/records/3541657/files/hobby.txt/content",
            "family.txt": "https://zenodo.org/api/records/3541657/files/family.txt/content"
        }

        downloaded_files = []

        for fname, url in file_links.items():
            print(f"Downloading {fname}...")
            r = requests.get(url, headers=headers, stream=True)
            if r.status_code == 200:
                file_path = os.path.join(output_dir, fname)
                with open(file_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                print(f"{fname} saved.")
                downloaded_files.append(file_path)
            else:
                print(f"Error: {fname}: {r.status_code}")

        return downloaded_files

    downloaded_user_files = download_user_files()
    return (downloaded_user_files,)


@app.cell
def _(downloaded_user_files, os):
    def preview_downloaded_files():

        for file_path in downloaded_user_files:
            if os.path.exists(file_path):
                fname = os.path.basename(file_path)
                print(f"\nFirst 5 lines {fname}:")
                with open(file_path, "r", encoding="utf-8") as f:
                    for i in range(5):
                        line = f.readline()
                        if not line:
                            break
                        print(line.strip())
            else:
                print(f"{file_path} does not exist")

    preview_downloaded_files()
    return


@app.cell
def _(os, pd, random):
    def create_aggregated_sample_dataset(input_dir="data/user_raw", output_csv="data/users_features.csv", sample_size=1000, random_seed=42):
        random.seed(random_seed)

        feature_files = ["gender.txt", "age.txt", "profession.txt", "hobby.txt", "family.txt"]
        feature_names = ["gender", "age", "profession", "hobby", "family"]

        family_path = os.path.join(input_dir, "family.txt")
        family_ids = []
        with open(family_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 4:
                    continue
                family_ids.append(parts[1])
        sampled_ids = set(random.sample(family_ids, sample_size))

        user_data = {uid: {feat: [] for feat in feature_names} for uid in sampled_ids}

        for fname, feature in zip(feature_files, feature_names):
            path = os.path.join(input_dir, fname)
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) < 4:
                        continue
                    uid = parts[1]
                    value = parts[2] 
                    if uid in sampled_ids:
                        user_data[uid][feature].append(value)

        rows = []
        for i, uid in enumerate(user_data.keys()):
            row = {"user_id": i+1} 
            for feat in feature_names:
                values = user_data[uid][feat]
                row[feat] = values[0] if values else None  
            rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(output_csv, index=False)
        return df

    sample_users_dataset = create_aggregated_sample_dataset("data/user_raw", sample_size=1000)
    print("First 10 rows of the aggregated sample dataset:")
    print(sample_users_dataset.head())

    complete_rows = sample_users_dataset.dropna().shape[0]
    print(f"\nNumber of complete rows (no missing features): {complete_rows} / {sample_users_dataset.shape[0]}")
    return (sample_users_dataset,)


@app.cell
def _(plt, sample_users_dataset, sns):
    # 02.2 EDA analysis
    def eda_analysis(df):
        plt.figure(figsize=(40, 24))
        sns.histplot(df['age'], bins=20, kde=True)
        plt.title('Age Distribution')
        plt.xlabel('Age')
        plt.ylabel('Frequency')
        plt.show()

        for col in ['hobby', 'profession']:
            print(f"\nUnique values in '{col}' ({df[col].nunique()}):")
            print(sorted(df[col].dropna().unique()))

        categorical_features = ['gender', 'profession', 'hobby', 'family']
        for feature in categorical_features:
            plt.figure(figsize=(50, 30))
            sns.countplot(
                y=feature,
                data=df,
                order=df[feature].value_counts().index
            )
            plt.title(f'{feature.capitalize()} Distribution')
            plt.xlabel('Count')
            plt.ylabel(feature.capitalize())
            plt.show()


    eda_analysis(sample_users_dataset)
    return


@app.cell
def _(np, pd):
    def clean_hobby_column(df, column="hobby"):
        remove = {
            'eating',
            'driving',
            'go',
            'learning',
            'entertaining',
            'people watching',
            'nan',
            'NaN',
            ''
        }

        mapping = {
            'gaming': 'video games',
            'games': 'video games',
            'video game': 'video games',
            'books': 'reading',
            'literature': 'reading',
            'dancing': 'dance',
            'golfing': 'golf',
            'crocheting': 'crochet',
            'traveling': 'travel',
            'rock climbing': 'climbing'
        }

        def process_hobby_cell(cell):
            if pd.isna(cell):
                return []

            hobbies = str(cell).split(":::")

            cleaned = []
            hobbies = str(cell).split(":::")
            cleaned = []
            for h in hobbies:
                h = h.strip().lower()
                h = mapping.get(h, h)
                if h not in remove:
                    cleaned.append(h)
            return cleaned if cleaned else np.nan

        df[column] = df[column].apply(process_hobby_cell)

        return df
    return (clean_hobby_column,)


@app.cell
def _(np, pd):
    def clean_profession_column(df, column="profession"):
        remove = {
            'brother', 'partner', 'employee', 'leader', 'ordinary', 'page', 'reader', 'organizer', 'earl', 'knight', 'prior', 'assassin', 'demon hunter', 'necromancer', 'ninja', 'rogue', 'psychic', 'sniper', 'slave', 'nazi', 'cyclist', 'runner', 'skier', 'surfer', 'swimmer', 'bodybuilder', 'gambler', 'motorcyclist', 'founder', 'nan', 'NaN', ''
        }

        mapping = {
            'physician': 'doctor',
            'registered nurse': 'nurse',
            'salesman': 'salesperson',
            'software engineer': 'software developer',
            'web developer': 'software developer',
            'graphic artist': 'graphic designer',
            'college professor': 'professor',
            'reporter': 'journalist',
            'columnist': 'journalist',
            'personal trainer': 'trainer',
            'chemist': 'scientist',
            'physicist': 'scientist',
            'microbiologist': 'scientist',
            'soil scientist': 'scientist',
            'investment banker': 'banker',
            'delivery driver': 'driver',
            'taxi driver': 'driver',
            'truck driver': 'driver'
        }

        def process_profession_cell(cell):
            if pd.isna(cell):
                return []

            professions = str(cell).split(":::")

            cleaned = []
            for p in professions:
                p = p.strip().lower()
                p = mapping.get(p, p)
                if p not in remove:
                    cleaned.append(p)
        
            return cleaned if cleaned else np.nan
            return cleaned

        df[column] = df[column].apply(process_profession_cell)
        return df
    return (clean_profession_column,)


@app.cell
def _(clean_hobby_column, clean_profession_column, sample_users_dataset):
    df_processed = sample_users_dataset.copy()
    df_processed = clean_hobby_column(df_processed)
    df_processed = clean_profession_column(df_processed)
    df_processed = df_processed.dropna()
    return (df_processed,)


@app.cell
def _(pd, plt, sns):
    def eda_analysis_lists(df):
        plt.figure(figsize=(40, 24))
        sns.histplot(df['age'], bins=20, kde=True)
        plt.title('Age Distribution')
        plt.xlabel('Age')
        plt.ylabel('Frequency')
        plt.show()

        for col in ['hobby', 'profession']:
            if isinstance(df[col].iloc[0], list):
                temp = df.explode(col)
            else:
                temp = df

            temp[col] = temp[col].astype(str)

            print(f"\nUnique values in '{col}' ({temp[col].nunique()}):")
            print(sorted(temp[col].dropna().unique()))

            order = pd.Series(temp[col]).value_counts().index.drop_duplicates()

            plt.figure(figsize=(50, 30))
            sns.countplot(
                y=col,
                data=temp,
                order=order
            )
            plt.title(f'{col.capitalize()} Distribution')
            plt.xlabel('Count')
            plt.ylabel(col.capitalize())
            plt.show()

        categorical_features = ['gender', 'family']
        for feature in categorical_features:
            plt.figure(figsize=(12, 6))
            sns.countplot(
                y=feature,
                data=df,
                order=df[feature].value_counts().index
            )
            plt.title(f'{feature.capitalize()} Distribution')
            plt.xlabel('Count')
            plt.ylabel(feature.capitalize())
            plt.show()
    return (eda_analysis_lists,)


@app.cell
def _(df_processed):
    df_processed.head(30)
    return


@app.cell
def _(df_processed, eda_analysis_lists):
    eda_analysis_lists(df_processed)
    return


@app.cell
def _(np, pd):
    # 02.3 Choose agents
    def choose_agents(df, target_size=100, hobby_col="hobby", profession_col="profession", random_state=None):
        if random_state is not None:
            np.random.seed(random_state)
    
        temp = df.copy()
        if isinstance(temp[hobby_col].iloc[0], list):
            temp = temp.explode(hobby_col)
    
        multi_hobby_mask = df[hobby_col].apply(lambda x: isinstance(x, list) and len(x) > 1)
        final_df = df[multi_hobby_mask].copy()
    
        remaining = df[~df.index.isin(final_df.index)].copy()
    
        professions_in_final = set(final_df[profession_col].dropna().apply(lambda x: x[0] if isinstance(x, list) else x))
    
        remaining = remaining.sample(frac=1, random_state=random_state)
    
        for idx, row in remaining.iterrows():
            if isinstance(row[profession_col], list):
                prof = row[profession_col][0]
            else:
                prof = row[profession_col]
        
            if prof not in professions_in_final:
                final_df = pd.concat([final_df, pd.DataFrame([row])])
                professions_in_final.add(prof)
        
            if len(final_df) >= target_size:
                break
    
        final_df = final_df.reset_index(drop=True)
    
        return final_df

    return (choose_agents,)


@app.cell
def _(choose_agents, df_processed):
    df_final = choose_agents(df_processed, target_size=50, random_state=42)
    print(df_final.shape)
    return (df_final,)


@app.cell
def _(df_final):
    df_final.head()
    return


@app.cell
def _(df_final):
    output_path = "data/users_features.csv"
    df_final.to_csv(output_path, index=False)

    print(f"df_final je spremljen u {output_path}")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
