# %%
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


# %%
def clean_text(text):
    # ciscenje teksta vijesti

    if pd.isna(text):
        return ""

    text = str(text)

    # uklanjanje URL-ova
    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text
    )

    # uklanjanje reuters oznake
    text = re.sub(
        r"\bReuters\b",
        " ",
        text,
        flags=re.IGNORECASE
    )

    # uklanjanje visestrukih razmaka
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()
# %%
def load_isot_data(fake_path, true_path):
    # ucitavanje Fake.csv i True.csv

    fake_df = pd.read_csv(fake_path)
    true_df = pd.read_csv(true_path)

    # 0 = real, 1 = fake
    true_df["label"] = 0
    fake_df["label"] = 1

    true_df["original_type"] = "real"
    fake_df["original_type"] = "fake"

    for frame in [true_df, fake_df]:

        frame["title"] = frame["title"].apply(
            clean_text
        )

        frame["text"] = frame["text"].apply(
            clean_text
        )

        frame["full_text"] = (
            frame["title"]
            + ". "
            + frame["text"]
        )

    columns = [
        "title",
        "text",
        "full_text",
        "label",
        "original_type"
    ]

    true_df = true_df[columns]
    fake_df = fake_df[columns]

    # spajanje real i fake vijesti
    data = pd.concat(
        [true_df, fake_df],
        ignore_index=True
    )

    # uklanjanje praznih i prekratkih tekstova
    data = data[
        data["full_text"].str.len() > 50
    ].copy()

    # uklanjanje duplikata
    data = data.drop_duplicates(
        subset=["full_text"]
    ).reset_index(drop=True)

    # jedinstveni ID svakog clanka
    data["source_id"] = [
        f"isot_{i:05d}"
        for i in range(len(data))
    ]

    return data
# %%
def split_dataset(
    data,
    test_size=0.20,
    random_state=42
):
    # podjela podataka na train i test skup

    train_df, test_df = train_test_split(
        data,
        test_size=test_size,
        stratify=data["label"],
        random_state=random_state
    )

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    return train_df, test_df
# %%
def save_splits(
    train_df,
    test_df,
    output_dir
):
    # spremanje train i test CSV datoteka

    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    train_df.to_csv(
        output_dir / "train.csv",
        index=False
    )

    test_df.to_csv(
        output_dir / "test.csv",
        index=False
    )
# %%
