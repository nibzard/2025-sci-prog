import marimo

__generated_with = "0.19.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md("""
    # Exploratory Data Analysis (EDA)
    ## Croatian Property Price Estimator

    Analiza ociscenih podataka za stanove i kuce.
    """)
    return


@app.cell
def _():
    import pandas as pd
    import numpy as np
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    return go, make_subplots, np, pd, px


@app.cell
def _(pd):
    # Ucitaj podatke
    df_kuce = pd.read_parquet("data/processed/houses_clean.parquet")
    df_stanovi = pd.read_parquet("data/processed/apartments_clean.parquet")
    return df_kuce, df_stanovi


@app.cell
def _(mo):
    mo.md("""
    ## 1. Pregled podataka
    """)
    return


@app.cell
def _(df_kuce, df_stanovi, mo):
    mo.md(f"""
    ### Velicina datasetova

    | Dataset | Broj redaka | Broj stupaca |
    |---------|-------------|--------------|
    | **Kuce** | {len(df_kuce):,} | {len(df_kuce.columns)} |
    | **Stanovi** | {len(df_stanovi):,} | {len(df_stanovi.columns)} |
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ### Stupci - Kuce
    """)
    return


@app.cell
def _(df_kuce):
    df_kuce.dtypes.to_frame("dtype")
    return


@app.cell
def _(mo):
    mo.md("""
    ### Stupci - Stanovi
    """)
    return


@app.cell
def _(df_stanovi):
    df_stanovi.dtypes.to_frame("dtype")
    return


@app.cell
def _(mo):
    mo.md("""
    ## 2. Statistike cijena
    """)
    return


@app.cell
def _(df_kuce, df_stanovi, pd):
    # Statistike cijena
    stats_kuce = df_kuce["cijena"].describe()
    stats_stanovi = df_stanovi["cijena"].describe()

    price_stats = pd.DataFrame({
        "Kuce": stats_kuce,
        "Stanovi": stats_stanovi
    }).round(2)
    price_stats
    return


@app.cell
def _(mo):
    mo.md("""
    ### Distribucija cijena - Kuce
    """)
    return


@app.cell
def _(df_kuce, px):
    fig_price_kuce = px.histogram(
        df_kuce,
        x="cijena",
        nbins=50,
        title="Distribucija cijena kuca",
        labels={"cijena": "Cijena (EUR)"},
        color_discrete_sequence=["#2E86AB"]
    )
    fig_price_kuce.update_layout(
        xaxis_tickformat=",",
        bargap=0.1
    )
    fig_price_kuce
    return


@app.cell
def _(mo):
    mo.md("""
    ### Distribucija cijena - Stanovi
    """)
    return


@app.cell
def _(df_stanovi, px):
    fig_price_stanovi = px.histogram(
        df_stanovi,
        x="cijena",
        nbins=50,
        title="Distribucija cijena stanova",
        labels={"cijena": "Cijena (EUR)"},
        color_discrete_sequence=["#A23B72"]
    )
    fig_price_stanovi.update_layout(
        xaxis_tickformat=",",
        bargap=0.1
    )
    fig_price_stanovi
    return


@app.cell
def _(mo):
    mo.md("""
    ### Box plot cijena (log skala)
    """)
    return


@app.cell
def _(df_kuce, df_stanovi, go, make_subplots, np):
    fig_box = make_subplots(rows=1, cols=2, subplot_titles=("Kuce", "Stanovi"))

    fig_box.add_trace(
        go.Box(y=np.log10(df_kuce["cijena"]), name="Kuce", marker_color="#2E86AB"),
        row=1, col=1
    )
    fig_box.add_trace(
        go.Box(y=np.log10(df_stanovi["cijena"]), name="Stanovi", marker_color="#A23B72"),
        row=1, col=2
    )

    fig_box.update_layout(
        title="Box plot cijena (log10 skala)",
        showlegend=False,
        height=400
    )
    fig_box.update_yaxes(title_text="log10(Cijena)", row=1, col=1)
    fig_box.update_yaxes(title_text="log10(Cijena)", row=1, col=2)
    fig_box
    return


@app.cell
def _(mo):
    mo.md("""
    ## 3. Cijena vs Povrsina
    """)
    return


@app.cell
def _(df_kuce, px):
    fig_scatter_kuce = px.scatter(
        df_kuce,
        x="stambena_povrsina",
        y="cijena",
        color="zupanija",
        title="Cijena vs Stambena povrsina - Kuce",
        labels={
            "stambena_povrsina": "Stambena povrsina (m2)",
            "cijena": "Cijena (EUR)"
        },
        hover_data=["grad_opcina", "naselje"],
        opacity=0.6
    )
    fig_scatter_kuce.update_layout(
        yaxis_tickformat=",",
        height=500
    )
    fig_scatter_kuce
    return


@app.cell
def _(df_stanovi, px):
    fig_scatter_stanovi = px.scatter(
        df_stanovi,
        x="stambena_povrsina",
        y="cijena",
        color="zupanija",
        title="Cijena vs Stambena povrsina - Stanovi",
        labels={
            "stambena_povrsina": "Stambena povrsina (m2)",
            "cijena": "Cijena (EUR)"
        },
        hover_data=["grad_opcina", "naselje"],
        opacity=0.6
    )
    fig_scatter_stanovi.update_layout(
        yaxis_tickformat=",",
        height=500
    )
    fig_scatter_stanovi
    return


@app.cell
def _(mo):
    mo.md("""
    ## 4. Analiza po lokaciji
    """)
    return


@app.cell
def _(df_kuce, px):
    # Prosjecna cijena po zupaniji - Kuce
    avg_price_zupanija_kuce = df_kuce.groupby("zupanija")["cijena"].agg(["mean", "count"]).reset_index()
    avg_price_zupanija_kuce.columns = ["zupanija", "prosjecna_cijena", "broj_oglasa"]
    avg_price_zupanija_kuce = avg_price_zupanija_kuce.sort_values("prosjecna_cijena", ascending=True)

    fig_zup_kuce = px.bar(
        avg_price_zupanija_kuce,
        x="prosjecna_cijena",
        y="zupanija",
        orientation="h",
        title="Prosjecna cijena kuca po zupaniji",
        labels={"prosjecna_cijena": "Prosjecna cijena (EUR)", "zupanija": "Zupanija"},
        color="broj_oglasa",
        color_continuous_scale="Blues",
        text="broj_oglasa"
    )
    fig_zup_kuce.update_layout(
        xaxis_tickformat=",",
        height=500
    )
    fig_zup_kuce.update_traces(texttemplate='%{text}', textposition='outside')
    fig_zup_kuce
    return


@app.cell
def _(df_stanovi, px):
    # Prosjecna cijena po zupaniji - Stanovi
    avg_price_zupanija_stanovi = df_stanovi.groupby("zupanija")["cijena"].agg(["mean", "count"]).reset_index()
    avg_price_zupanija_stanovi.columns = ["zupanija", "prosjecna_cijena", "broj_oglasa"]
    avg_price_zupanija_stanovi = avg_price_zupanija_stanovi.sort_values("prosjecna_cijena", ascending=True)

    fig_zup_stanovi = px.bar(
        avg_price_zupanija_stanovi,
        x="prosjecna_cijena",
        y="zupanija",
        orientation="h",
        title="Prosjecna cijena stanova po zupaniji",
        labels={"prosjecna_cijena": "Prosjecna cijena (EUR)", "zupanija": "Zupanija"},
        color="broj_oglasa",
        color_continuous_scale="RdPu",
        text="broj_oglasa"
    )
    fig_zup_stanovi.update_layout(
        xaxis_tickformat=",",
        height=500
    )
    fig_zup_stanovi.update_traces(texttemplate='%{text}', textposition='outside')
    fig_zup_stanovi
    return


@app.cell
def _(mo):
    mo.md("""
    ## 5. Missing Values analiza
    """)
    return


@app.cell
def _(df_kuce, px):
    # Missing values - Kuce
    missing_kuce = df_kuce.isnull().sum()
    missing_kuce = missing_kuce[missing_kuce > 0].sort_values(ascending=True)
    missing_kuce_pct = (missing_kuce / len(df_kuce) * 100).round(1)

    fig_missing_kuce = px.bar(
        x=missing_kuce_pct.values,
        y=missing_kuce_pct.index,
        orientation="h",
        title="Missing values - Kuce (%)",
        labels={"x": "Postotak missing", "y": "Stupac"},
        color=missing_kuce_pct.values,
        color_continuous_scale="Reds"
    )
    fig_missing_kuce.update_layout(height=400)
    fig_missing_kuce
    return


@app.cell
def _(df_stanovi, px):
    # Missing values - Stanovi
    missing_stanovi = df_stanovi.isnull().sum()
    missing_stanovi = missing_stanovi[missing_stanovi > 0].sort_values(ascending=True)
    missing_stanovi_pct = (missing_stanovi / len(df_stanovi) * 100).round(1)

    fig_missing_stanovi = px.bar(
        x=missing_stanovi_pct.values,
        y=missing_stanovi_pct.index,
        orientation="h",
        title="Missing values - Stanovi (%)",
        labels={"x": "Postotak missing", "y": "Stupac"},
        color=missing_stanovi_pct.values,
        color_continuous_scale="Reds"
    )
    fig_missing_stanovi.update_layout(height=400)
    fig_missing_stanovi
    return


@app.cell
def _(mo):
    mo.md("""
    ## 6. Korelacija numerickih varijabli
    """)
    return


@app.cell
def _(df_kuce, px):
    # Korelacijska matrica - Kuce
    numeric_cols_kuce = df_kuce.select_dtypes(include=["float64", "int64"]).columns
    corr_kuce = df_kuce[numeric_cols_kuce].corr()

    # Korelacija s cijenom
    corr_with_price_kuce = corr_kuce["cijena"].drop("cijena").sort_values(ascending=False)

    fig_corr_kuce = px.bar(
        x=corr_with_price_kuce.values,
        y=corr_with_price_kuce.index,
        orientation="h",
        title="Korelacija s cijenom - Kuce",
        labels={"x": "Korelacija", "y": "Varijabla"},
        color=corr_with_price_kuce.values,
        color_continuous_scale="RdBu_r",
        range_color=[-1, 1]
    )
    fig_corr_kuce.update_layout(height=600)
    fig_corr_kuce
    return


@app.cell
def _(df_stanovi, px):
    # Korelacijska matrica - Stanovi
    numeric_cols_stanovi = df_stanovi.select_dtypes(include=["float64", "int64"]).columns
    corr_stanovi = df_stanovi[numeric_cols_stanovi].corr()

    # Korelacija s cijenom
    corr_with_price_stanovi = corr_stanovi["cijena"].drop("cijena").sort_values(ascending=False)

    fig_corr_stanovi = px.bar(
        x=corr_with_price_stanovi.values,
        y=corr_with_price_stanovi.index,
        orientation="h",
        title="Korelacija s cijenom - Stanovi",
        labels={"x": "Korelacija", "y": "Varijabla"},
        color=corr_with_price_stanovi.values,
        color_continuous_scale="RdBu_r",
        range_color=[-1, 1]
    )
    fig_corr_stanovi.update_layout(height=600)
    fig_corr_stanovi
    return


@app.cell
def _(mo):
    mo.md("""
    ## 7. Analiza kategorickih varijabli
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ### Tip kuce
    """)
    return


@app.cell
def _(df_kuce, px):
    # Tip kuce - distribucija
    tip_kuce_cols = [c for c in df_kuce.columns if c.startswith("tip_kuce_")]
    tip_kuce_counts = df_kuce[tip_kuce_cols].sum().sort_values(ascending=True)
    tip_kuce_counts.index = [c.replace("tip_kuce_", "").replace("_", " ").title() for c in tip_kuce_counts.index]

    fig_tip_kuce = px.bar(
        x=tip_kuce_counts.values,
        y=tip_kuce_counts.index,
        orientation="h",
        title="Distribucija tipova kuca",
        labels={"x": "Broj", "y": "Tip kuce"},
        color_discrete_sequence=["#2E86AB"]
    )
    fig_tip_kuce
    return


@app.cell
def _(mo):
    mo.md("""
    ### Tip stana
    """)
    return


@app.cell
def _(df_stanovi, px):
    # Tip stana - distribucija
    tip_stana_cols = [c for c in df_stanovi.columns if c.startswith("tip_stana_")]
    tip_stana_counts = df_stanovi[tip_stana_cols].sum().sort_values(ascending=True)
    tip_stana_counts.index = [c.replace("tip_stana_", "").replace("_", " ").title() for c in tip_stana_counts.index]

    fig_tip_stana = px.bar(
        x=tip_stana_counts.values,
        y=tip_stana_counts.index,
        orientation="h",
        title="Distribucija tipova stanova",
        labels={"x": "Broj", "y": "Tip stana"},
        color_discrete_sequence=["#A23B72"]
    )
    fig_tip_stana
    return


@app.cell
def _(mo):
    mo.md("""
    ### Energetski razred
    """)
    return


@app.cell
def _(df_kuce, df_stanovi, go, make_subplots):
    # Energetski razred distribucija
    energy_labels = {5: "A+", 4: "A", 3: "B", 2: "C", 1: "D", 0: "E", -1: "F", -2: "G"}

    energy_kuce = df_kuce["energetski_razred"].dropna().value_counts().sort_index()
    energy_stanovi = df_stanovi["energetski_razred"].dropna().value_counts().sort_index()

    fig_energy = make_subplots(rows=1, cols=2, subplot_titles=("Kuce", "Stanovi"))

    fig_energy.add_trace(
        go.Bar(
            x=[energy_labels.get(i, str(i)) for i in energy_kuce.index],
            y=energy_kuce.values,
            marker_color="#2E86AB",
            name="Kuce"
        ),
        row=1, col=1
    )

    fig_energy.add_trace(
        go.Bar(
            x=[energy_labels.get(i, str(i)) for i in energy_stanovi.index],
            y=energy_stanovi.values,
            marker_color="#A23B72",
            name="Stanovi"
        ),
        row=1, col=2
    )

    fig_energy.update_layout(
        title="Distribucija energetskih razreda",
        showlegend=False,
        height=400
    )
    fig_energy
    return


@app.cell
def _(mo):
    mo.md("""
    ### Broj soba
    """)
    return


@app.cell
def _(df_kuce, df_stanovi, go, make_subplots):
    fig_sobe = make_subplots(rows=1, cols=2, subplot_titles=("Kuce", "Stanovi"))

    sobe_kuce = df_kuce["broj_soba"].value_counts().sort_index()
    sobe_stanovi = df_stanovi["broj_soba"].value_counts().sort_index()

    fig_sobe.add_trace(
        go.Bar(x=sobe_kuce.index, y=sobe_kuce.values, marker_color="#2E86AB", name="Kuce"),
        row=1, col=1
    )

    fig_sobe.add_trace(
        go.Bar(x=sobe_stanovi.index, y=sobe_stanovi.values, marker_color="#A23B72", name="Stanovi"),
        row=1, col=2
    )

    fig_sobe.update_layout(
        title="Distribucija broja soba",
        showlegend=False,
        height=400
    )
    fig_sobe.update_xaxes(title_text="Broj soba", row=1, col=1)
    fig_sobe.update_xaxes(title_text="Broj soba", row=1, col=2)
    fig_sobe
    return


@app.cell
def _(mo):
    mo.md("""
    ## 8. Cijena po m2
    """)
    return


@app.cell
def _(df_kuce, df_stanovi, go, make_subplots, np):
    # Izracunaj cijenu po m2
    cijena_m2_kuce = df_kuce["cijena"] / df_kuce["stambena_povrsina"]
    cijena_m2_stanovi = df_stanovi["cijena"] / df_stanovi["stambena_povrsina"]

    fig_m2 = make_subplots(rows=1, cols=2, subplot_titles=("Kuce", "Stanovi"))

    fig_m2.add_trace(
        go.Histogram(x=cijena_m2_kuce, nbinsx=50, marker_color="#2E86AB", name="Kuce"),
        row=1, col=1
    )

    fig_m2.add_trace(
        go.Histogram(x=cijena_m2_stanovi, nbinsx=50, marker_color="#A23B72", name="Stanovi"),
        row=1, col=2
    )

    fig_m2.update_layout(
        title="Distribucija cijene po m2",
        showlegend=False,
        height=400
    )
    fig_m2.update_xaxes(title_text="EUR/m2", row=1, col=1, range=[0, np.percentile(cijena_m2_kuce.dropna(), 99)])
    fig_m2.update_xaxes(title_text="EUR/m2", row=1, col=2, range=[0, np.percentile(cijena_m2_stanovi.dropna(), 99)])
    fig_m2
    return cijena_m2_kuce, cijena_m2_stanovi


@app.cell
def _(cijena_m2_kuce, cijena_m2_stanovi, mo):
    mo.md(f"""
    ### Statistike cijene po m2

    | Metrika | Kuce | Stanovi |
    |---------|------|---------|
    | **Prosjek** | {cijena_m2_kuce.mean():,.0f} EUR/m2 | {cijena_m2_stanovi.mean():,.0f} EUR/m2 |
    | **Medijan** | {cijena_m2_kuce.median():,.0f} EUR/m2 | {cijena_m2_stanovi.median():,.0f} EUR/m2 |
    | **Min** | {cijena_m2_kuce.min():,.0f} EUR/m2 | {cijena_m2_stanovi.min():,.0f} EUR/m2 |
    | **Max** | {cijena_m2_kuce.max():,.0f} EUR/m2 | {cijena_m2_stanovi.max():,.0f} EUR/m2 |
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 9. Godina izgradnje
    """)
    return


@app.cell
def _(df_kuce, df_stanovi, go, make_subplots):
    fig_godina = make_subplots(rows=1, cols=2, subplot_titles=("Kuce", "Stanovi"))

    godina_kuce = df_kuce["godina_izgradnje"].dropna()
    godina_stanovi = df_stanovi["godina_izgradnje"].dropna()

    fig_godina.add_trace(
        go.Histogram(x=godina_kuce, nbinsx=30, marker_color="#2E86AB", name="Kuce"),
        row=1, col=1
    )

    fig_godina.add_trace(
        go.Histogram(x=godina_stanovi, nbinsx=30, marker_color="#A23B72", name="Stanovi"),
        row=1, col=2
    )

    fig_godina.update_layout(
        title="Distribucija godine izgradnje",
        showlegend=False,
        height=400
    )
    fig_godina.update_xaxes(title_text="Godina", row=1, col=1)
    fig_godina.update_xaxes(title_text="Godina", row=1, col=2)
    fig_godina
    return


@app.cell
def _(mo):
    mo.md("""
    ## 10. Zakljucci

    ### Kljucna zapazanja:

    1. **Cijene**: Kuce imaju veci raspon cijena i vecu prosjecnu cijenu od stanova
    2. **Lokacija**: Zupanije na obali (Primorsko-goranska, Splitsko-dalmatinska) imaju najvece cijene
    3. **Missing values**: Energetski razred i broj WC-a imaju najvise nedostajucih vrijednosti
    4. **Povrsina**: Postoji jaka korelacija izmedu stambene povrsine i cijene
    5. **Novogradnja**: Velik broj stanova je novogradnja (2024-2027)
    """)
    return


if __name__ == "__main__":
    app.run()
