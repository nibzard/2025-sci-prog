import marimo

__generated_with = "0.19.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    <div style="text-align: center; font-size: 50px">
        PROJEKT
    </div>
    <div style="text-align: center; font-size: 34px;">
        Mario Puizina<br>
        Znanstveno programiranje 2025./2026.
    </div>
    """)
    return


@app.cell
def _(mo):
    mo.md(f"""
    # Problemi u pčelarstvu

    Komercijalno pčelarstvo se trenutno suočava s velikom krizom izumiranja kolonija.
    Anketa koja je nedavno provedena diljem SAD-a otkriva porazne brojke:

    [Survey Reveals Over 1.1 Million Honey Bee Colonies Lost, Raising Alarm for Pollination and Agriculture](https://honeybeehealthcoalition.org/survey-reveals-over-1-1-million-honey-bee-colonies-lost-raising-alarm-for-pollination-and-agriculture/)

    > „A nationwide survey of beekeepers has revealed catastrophic honey bee colony losses across the United States, with commercial operations reporting an **average loss of 62%** between June 2024 and February 2025.”

    {mo.image(src="assets/pcela.jpg", alt="Pčela", width=400)}
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    # Negativni utjecaji na zdravlje kolonija

    - **Paraziti i patogeni (virusi, gljivice)**
    - **Pesticidi**
        - Jedan pesticid sam po sebi ne mora imati loš utjecaj, ali više pesticida zajedno mogu imati potentniji sinergistički efekt i tako naštetiti izloženim pčelama.
    - **Niska bioraznolikost za oprašivanje**
        - Pčele se često koriste za oprašivanje monokultura poput badema gdje
        ne dobivaju dovoljno raznolike nutrijente. Zbog toga njihov imunosni sustav može ispaštati što ih čini ranjivijima na već spomenute parazite i patogene.
    - **Nestabilna klima**
        - Nepredvidive vremenske promjene mogu uzrokovati prerano ili prekasno cvjetanje biljaka zbog čega pčele mogu biti nesinkronizirane sa svojom okolinom u vrijeme oprašivanja i posljedično nemati dovoljno hrane.
    - **...**
    """)
    return


@app.cell
def _(mo):
    mo.md(f"""
    # _Varroa destructor_

    - _Varroa destructor_ je parazit koji pčelarstvu nanosi najveću ekonomsku štetu
    - Hrani se organom pčele koji se naziva "masno tijelo" (_fat body_) i tako oslabljuje pčelin imunosni sustav
    - Vektor je za viruse poput DWV (_Deformed Wing Virus_)
    - Razmnožava se unutar košnice u poklopljenim leglima još nerazvijenih pčela
    - Ako je 3% pčela zaraženo, potreban je tretman (miticidi) jer će u suprotnom previše pčela radilica umrijeti i kolonija neće opstati
    - Problem pretjeranog korištenja miticida je taj što su _Varroa_ paraziti zahvaljujući svom brzom razmnožavanju i izmjenama generacija počeli razvijati genetsku otpornost na postojeća rješenja

    {mo.hstack([
        mo.image(src="assets/varroa.jpg", alt="Varroa", width="230px"),
        mo.image(src="assets/bee_with_varroa.jpeg", alt="Bee with Varroa", width="550px")
    ], justify="start")}
    """)
    return


@app.cell
def _(mo):
    mo.md(f"""
    <div style="text-align: center; font-size: 50px">
        ZNANSTVENI RAD
    </div>
    <div style="text-align: center; font-size: 34px;">
        <a href="https://journals.plos.org/climate/article?id=10.1371/journal.pclm.0000485", style="color: black; text-decoration: underline;">
            <i>Climatic predictors of prominent honey bee disease agents</i>
        </a>
    </div>
    """)
    return


@app.cell
def _(mo):
    mo.md(f"""
    Autori ovog znanstvenog rada su istraživali utjecaj vanjskih čimbenika (vremenski uvjeti, geografska lokacija i period godine) na prisutnost _Varroa destructor_ i drugih štetnika _(Melissococcus plutonius, Vairimorpha)_ u kolonijama pčela.

    Fokus ovog projekta će biti na dijelu istraživanja u kojem su istraživači proveli statističku analizu nad podatcima iz 2020. godine tijekom koje se pratila prisutnost _Varroa destructor_ u 240 kolonija pčela rasprostranjenih na 6 lokacija unutar Kanade. Ovo nam je omogućeno zato što su autori istraživanja uz svoj znanstveni rad javno objavili podatke i R kod s kojim su proveli analizu.

    {mo.image(src="assets/kolonije.png", alt="Kolonije u Kanadi")}
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    Učitajmo podatke...
    """)
    return


@app.cell
def _():
    import pandas as pd

    raw_data2020 = pd.read_csv("bee_colonies_and_varroa_mites_2020.csv")
    raw_data2020.head()
    return pd, raw_data2020


@app.cell
def _(mo, raw_data2020):
    mo.md(f"""
    Proučimo prvih 5 stupaca:

    - `Colony.label` [{raw_data2020['Colony.label'].dtype}] - identifikator kolonije
        - Broj jedinstvenih kolonija: {raw_data2020['Colony.label'].nunique()}
    - `Crop` [{raw_data2020['Crop'].dtype}] - autori istraživanja su originalno htjeli promatrati utjecaj obližnjih usjeva na zdravlje kolonija
        - {raw_data2020['Crop'].unique()}
    - `Province` [{raw_data2020['Province'].dtype}] - kanadska provincija u kojoj se kolonija nalazi
        - {raw_data2020['Province'].unique()}
    - `Location` [{raw_data2020['Location'].dtype}] - preciznija lokacija kolonije
        - {raw_data2020['Location'].unique()}
    - `Site.type` [{raw_data2020['Site.type'].dtype}] - još jedna značajka vezana za usjeve; označava je li kolonija bila daleko (_unexposed_) ili blizu usjeva (_exposed_)
        - {raw_data2020['Site.type'].unique()}
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    Budući na to da nećemo gledati utjecaj obližnjih usjeva, uklonit ćemo stupce `Crop` i `Site.type`.

    Također, stupac `Location` će nam biti sasvim dovoljan za istraživanje regionalnih utjecaja tako da ćemo ukloniti i redundantan stupac `Province`.
    """)
    return


@app.cell
def _(mo):
    first_columns_to_drop = ['Crop', 'Site.type', 'Province']

    mo.md(f"Stupci koji će biti uklonjeni: {first_columns_to_drop}")
    return (first_columns_to_drop,)


@app.cell
def _(mo):
    mo.md("""
    Pogledajmo kako su kolonije raspoređene po lokacijama:
    """)
    return


@app.cell
def _(raw_data2020):
    import matplotlib.pyplot as plt

    def plot_colonies():
        plot_df = raw_data2020[['Colony.label', 'Location']].copy()

        # Calculate the number of unique colonies per location
        colony_counts_per_location = (
            plot_df.groupby('Location', observed=True)['Colony.label']
            .nunique()
            .sort_values(ascending=False)
        )

        fig, ax = plt.subplots(figsize=(10, 6))

        colony_counts_per_location.plot(kind='bar', color='skyblue', edgecolor='black', ax=ax)

        # Adding the labels
        ax.bar_label(ax.containers[0], padding=3)

        ax.set_title('Broj kolonija po lokaciji', fontsize=14)
        ax.set_xlabel('Lokacija', fontsize=12)
        ax.set_ylabel('Broj kolonija', fontsize=12)
        ax.tick_params(axis='x', rotation=0)
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        plt.tight_layout()

        # Return the figure object so Marimo can display it
        return fig

    # This is the line that makes it show up in App Mode
    plot_colonies()
    return (plt,)


@app.cell
def _(mo, raw_data2020):
    mo.md(f"""
    Prijeđimo na sljedećih 5 stupaca:

    - `Time.point` [{raw_data2020['Time.point'].dtype}]
    - `Date` [{raw_data2020['Date'].dtype}]
    - `Total.bees.in.sample (based on weight)` [{raw_data2020['Total.bees.in.sample (based on weight)'].dtype}]
    - `Total.mites (counts by alcohol wash)` [{raw_data2020['Total.mites (counts by alcohol wash)'].dtype}]
    - `Percent.mites` [{raw_data2020['Percent.mites'].dtype}]

    Ovi stupci su vezani za tzv. _alcohol wash_ tehniku. To je tehnika kojom se provjerava kolika je neka kolonija pčela zaražena _Varroa_ parazitom. Određeni broj pčela (cilj je oko ~300 pčela na osnovu gramaže) promiješa se u posudi s alkoholom nakon čega eventualni _Varroa_ paraziti koji su prisutni na pčelama padnu kroz filter na dno posude kako bi ih se moglo prebrojati. Pčele u posudi nažalost umru, ali ovaj postupak je potreban za dobrobit kolonije koja se sastoji od više desetaka tisuća pčela.

    {mo.image(src="assets/alcohol_wash.webp", alt="Alcohol Wash", width="250px")}
    """)
    return


@app.cell
def _(mo):
    mo.md(f"""
    Stupac `Time.point` se odnosi na 3 različita _alcohol wash_ postupka koja su provedena nad promatranim kolonijama tijekom 2020. godine. Prvi postupak je proveden prije perioda oprašivanja, drugi na vrhuncu perioda oprašivanja, a treći na kraju perioda oprašivanja.

    Vrijednosti u stupcu `Time.point`:
    """)
    return


@app.cell
def _(pd, raw_data2020):
    def get_timepoint_summary():
        mapping = {
            't1': 'Test prije perioda oprašivanja',
            't2': 'Test na vrhuncu perioda oprašivanja',
            't3': 'Test poslije perioda oprašivanja'
        }

        summary = pd.DataFrame({
            'Time Point': sorted(raw_data2020['Time.point'].unique())
        })

        summary['Opis'] = summary['Time Point'].map(mapping)

        return summary

    get_timepoint_summary()
    return


@app.cell
def _(mo):
    mo.md(r"""
    Pogledajmo koliko je kolonija testirano 1, 2, odnosno 3 puta...
    """)
    return


@app.cell
def _(raw_data2020):
    def summarize_colonies_per_number_of_tests():
        local_df = raw_data2020[['Colony.label', 'Time.point']]

        # 1. Count unique time points per colony
        colony_counts = local_df.groupby('Colony.label')['Time.point'].nunique()

        # 2. Aggregate counts and reindex
        summary_df = (
            colony_counts.value_counts()
            .reindex([1, 2, 3], fill_value=0)
            .reset_index()
        )

        # 3. Rename columns and format the 'Times Tested' strings
        summary_df.columns = ['Number of Tests', 'Number of colonies']

        # Use an f-string via .apply() or .map() to add the suffix
        summary_df['Number of Tests'] = summary_df['Number of Tests'].apply(lambda x: f"{x} tests")
        return summary_df

    summarize_colonies_per_number_of_tests()
    return


@app.cell
def _(mo, raw_data2020):
    mo.md(f"""
    Dakle, 240 kolonija pčela je testirano po 3 puta što znači da bi trebali imati 720 redaka u datasetu.

    Broj redaka: {raw_data2020.shape[0]}
    """)
    return


@app.cell
def _(mo, raw_data2020):
    mo.md(f"""
    Stupac `Date` se odnosi na konkretne datume kada su kolonije testirane.

    Broj nedostajućih vrijednosti u stupcu `Date`: {raw_data2020['Date'].isna().sum()}

    Raspon vrijednosti u stupcu `Date`: [{raw_data2020['Date'].min()} ... {raw_data2020['Date'].max()}]

    Stupac `Date` je u neobičnom formatu, pogledajmo tip podatka:

    {raw_data2020['Date'].dtype}

    Tip podatka je ovakav zato što je `Date` pohranjen kao _Excel serial date_ kojim se datum označava kao broj dana koji je prošao od 30.12.1899.

    Izvršavamo konverziju u `datetime[ns]`, stvaramo stupac `date_formatted` i ponovno ispisujemo raspon datuma...
    """)
    return


@app.cell
def _(mo, pd, raw_data2020):
    raw_data2020['date_formatted'] = pd.to_datetime(raw_data2020['Date'], unit='D', origin='1899-12-30')

    min_date = raw_data2020['date_formatted'].min()
    max_date = raw_data2020['date_formatted'].max()

    mo.md(f"Raspon vrijednosti u stupcu `date_formatted`: [{min_date.date()} ... {max_date.date()}]")
    return


@app.cell
def _(mo, raw_data2020):
    def check_time_point_temporal_order():
        # 1. Pivot so each Time.point is a column
        # index: Colony ID | columns: t1, t2, t3 | values: the dates
        validation_df = raw_data2020.pivot(
            index='Colony.label', 
            columns='Time.point', 
            values='date_formatted'
        )

        # 2. Check the chronological logic across columns
        # We want t1 < t2 and t2 < t3
        is_chronological = (validation_df['t1'] < validation_df['t2']) & \
                           (validation_df['t2'] < validation_df['t3'])

        return is_chronological.all()

    mo.md(f"""
    Vrijedi li `t3` > `t2` > `t1` za sve kolonije?

    {check_time_point_temporal_order()}
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    Promatrane kolonije nisu nužno za svaki `Time.point` testirane na isti datum. Razlog tome je što periodi oprašivanja mogu varirati između lokacija.

    Kako bi dobili osjećaj o kakvim datumima je riječ, pogledajmo periode unutar kojih su testiranja provedena.
    """)
    return


@app.cell
def _(pd, plt, raw_data2020):
    import matplotlib.dates as mdates

    def plot_sampling_periods():
        plot_df = raw_data2020[['Time.point', 'date_formatted']].copy()

        # 1. Map labels
        plot_df['Sampling Period'] = plot_df['Time.point'].map({
            't1': 'Prije oprašivanja',
            't2': 'Vrhunac oprašivanja',
            't3': 'Poslije oprašivanja'
        })

        # 2. Aggregation
        spans = plot_df.groupby('Sampling Period')['date_formatted'].agg(['min', 'max']).reset_index()

        # 3. Ordering
        chronological_order = ['Prije oprašivanja', 'Vrhunac oprašivanja', 'Poslije oprašivanja']
        spans['Sampling Period'] = pd.Categorical(spans['Sampling Period'], categories=chronological_order, ordered=True)
        spans = spans.sort_values('Sampling Period', ascending=False)

        fig, ax = plt.subplots(figsize=(12, 5))

        # 4. Plotting with distinct edge colors
        for i, (idx, row) in enumerate(spans.iterrows()):
            # Draw the connecting line
            ax.hlines(y=row['Sampling Period'], xmin=row['min'], xmax=row['max'], 
                      color='skyblue', linewidth=10, alpha=0.5)

            # Plot start point (Min) - Label only the first iteration for the legend
            ax.scatter(row['min'], row['Sampling Period'], color='crimson', s=80, zorder=3,
                       label='Najraniji test' if i == 0 else "")

            # Plot end point (Max) - Label only the first iteration for the legend
            ax.scatter(row['max'], row['Sampling Period'], color='green', s=80, zorder=3,
                       label='Najkasniji test' if i == 0 else "")

        # 5. Formatting and Legend
        ax.set_title('Vremenski rasponi faza testiranja (od "najranijih" lokacija do "najkasnijih")', fontsize=14, pad=20)
        ax.set_xlabel('Datum', fontsize=12)
        ax.set_ylabel('Period', fontsize=12)

        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.MonthLocator())

        # Add the legend
        ax.legend(loc='upper right', frameon=True, shadow=True)

        ax.grid(axis='x', linestyle='--', alpha=0.5)
        plt.tight_layout()
        return fig

    plot_sampling_periods()
    return


@app.cell
def _(mo):
    mo.md("""
    Čisto iz znatiželje, pogledajmo na kojoj je lokaciji testiranje u prosjeku počelo najranije, a na kojoj najkasnije (dakle, gledamo prosječni datum prvog testa `t1`):
    """)
    return


@app.cell
def _(mo, raw_data2020):
    def find_locations_with_earliest_and_latest_first_test():
        t1_data = raw_data2020[raw_data2020['Time.point'] == 't1']

        avg_dates = t1_data.groupby('Location')['date_formatted'].mean()

        # Format the dates to ISO 8601 (YYYY-MM-DD)
        earliest_date = avg_dates.min().strftime('%Y-%m-%d')
        latest_date = avg_dates.max().strftime('%Y-%m-%d')

        return mo.md(f"""
            Najraniji prosječni početak testiranja: {avg_dates.idxmin()} [{earliest_date}]

            Najkasniji prosječni početak testiranja: {avg_dates.idxmax()} [{latest_date}]
        """)

    find_locations_with_earliest_and_latest_first_test()
    return


@app.cell
def _(mo, raw_data2020):
    mo.md(f"""
    Proučimo sada ova 3 stupca koji sadrže podatke s provedenih _alcohol wash_ postupaka:

    - `Total.bees.in.sample (based on weight)` [{raw_data2020['Total.bees.in.sample (based on weight)'].dtype}] - procijenjen broj pčela u posudi na osnovu gramaže
        - Broj nedostajućih vrijednosti: {raw_data2020['Total.bees.in.sample (based on weight)'].isna().sum()}
        - Prosječan broj testiranih pčela: {raw_data2020['Total.bees.in.sample (based on weight)'].mean():.2f}

    - `Total.mites (counts by alcohol wash)` [{raw_data2020['Total.mites (counts by alcohol wash)'].dtype}] - broj _Varroa destructor_ parazita na dnu posude
        - Broj nedostajućih vrijednosti: {raw_data2020['Total.mites (counts by alcohol wash)'].isna().sum()}
        - Raspon vrijednosti: [{raw_data2020['Total.mites (counts by alcohol wash)'].min()} - {raw_data2020['Total.mites (counts by alcohol wash)'].max()}]
        - Prosječan broj parazita: {raw_data2020['Total.mites (counts by alcohol wash)'].mean():.2f}

    - `Percent.mites` [{raw_data2020['Percent.mites'].dtype}] - broj parazita na 100 pčela izražen kao postotak
        - Broj nedostajućih vrijednosti: {raw_data2020['Percent.mites'].isna().sum()}
        - Raspon vrijednosti: [{raw_data2020['Percent.mites'].min()}% - {raw_data2020['Percent.mites'].max():.2f}%]
        - Prosječan postotak: {raw_data2020['Percent.mites'].mean():.2f}%
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    S obzirom na to da ćemo se u sklopu ovog projekta isključivo fokusirati na _prisutnost_, a ne _intenzitet_ zaraze parazitom _Varroa destructor_, uklonit ćemo stupce `Total.bees.in.sample (based on weight)` i `Total.mites (counts by alcohol wash)`.
    """)
    return


@app.cell
def _(first_columns_to_drop, mo):
    second_columns_to_drop = [
        'Total.bees.in.sample (based on weight)',
        'Total.mites (counts by alcohol wash)'
    ]
    mo.md(f"""
    Stupci koji će se ukloniti:

    {first_columns_to_drop + second_columns_to_drop}
    """)
    return (second_columns_to_drop,)


@app.cell
def _(mo):
    mo.md("""
    Stupac `Percent.mites` ćemo ostaviti za kratku eksplorativnu analizu i kasnije stvaranje binarne zavisne varijable koja će označavati je li kolonija zaražena ili ne.

    Budući da `Percent.mites` ima 4 nedostajuće vrijednosti, kasnije ćemo ukloniti sljedeća 4 retka:
    """)
    return


@app.cell
def _(first_columns_to_drop, raw_data2020, second_columns_to_drop):
    first_rows_to_drop = raw_data2020[raw_data2020['Percent.mites'].isna()].drop(
        columns=first_columns_to_drop + second_columns_to_drop
    )

    first_rows_to_drop
    return (first_rows_to_drop,)


@app.cell
def _(first_rows_to_drop, raw_data2020):
    raw_data_minus_4 = raw_data2020.drop(index=first_rows_to_drop.index).copy()
    return (raw_data_minus_4,)


@app.cell
def _(mo):
    mo.md("""
    Pogledajmo proporcije zaraženih i nezaraženih uzoraka.
    """)
    return


@app.cell
def _(plt, raw_data_minus_4):
    import numpy as np
    import seaborn as sns

    def plot_mite_presence() -> plt.Figure:
        # 1. Prepare the data
        plot_df = raw_data_minus_4.copy()
        plot_df['Infestation_Status'] = np.where(
            plot_df['Percent.mites'] > 0, 'Zaražen', 'Nezaražen'
        )

        # 2. Initialize the figure and axis
        # In Marimo, explicitly creating the figure object is best for returning it
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.set_theme(style="whitegrid")

        # 3. Create the count plot
        sns.countplot(
            data=plot_df, 
            x='Infestation_Status', 
            order=['Nezaražen', 'Zaražen'],
            palette=['#3498db', '#e74c3c'],
            hue='Infestation_Status',
            legend=False,
            ax=ax
        )

        # 4. Academic formatting with LaTeX
        ax.set_title(r'Nezaraženi i zaraženi uzorci s $\mathit{Varroa\ destructor}$', fontsize=14)
        ax.set_xlabel('Status', fontsize=12)
        ax.set_ylabel('Broj uzoraka', fontsize=12)

        # Add count labels on top of bars
        for p in ax.patches:
            ax.annotate(f'{int(p.get_height())}', 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha='center', va='center', 
                        xytext=(0, 9), 
                        textcoords='offset points',
                        fontsize=11)

        plt.tight_layout()

        return fig

    plot_mite_presence()
    return np, sns


@app.cell
def _(mo):
    mo.md("""
    Možemo odmah primijetiti da je skup podataka dominiran nulama, odnosno uzorcima gdje _Varroa_ nije pronađena.

    Pogledajmo još kako su distribuirani postotci prisutnih parazita u slučajevima gdje je došlo do zaraze.
    """)
    return


@app.cell
def _(plt, raw_data_minus_4, sns):
    def plot_mite_infestation() -> plt.Figure:
        # Filter for samples where mites are present (> 0)
        present_mites = raw_data_minus_4[raw_data_minus_4['Percent.mites'] > 0]

        # Initialize the figure
        fig, ax = plt.subplots(figsize=(8, 6))

        # Create the boxplot
        sns.boxplot(data=present_mites, y='Percent.mites', ax=ax, color='skyblue')

        # Adding a swarmplot on top can be useful for DS analysis to see sample density
        sns.swarmplot(data=present_mites, y='Percent.mites', ax=ax, color='black', alpha=0.5)

        # Formatting
        ax.set_title('Distribucija Varroa destructor postotaka (zaraženi uzorci)')
        ax.set_ylabel('Broj parazita po 100 pčela (%)')

        return fig

    plot_mite_infestation()
    return


@app.cell
def _(mo, raw_data2020):
    mo.md(f"""
    Proučimo još posljednja 4 stupca koja opisuju vremenske uvjete u blizini kolonije 3 tjedna prije svakog uzorka:

    - `Mean.temp.3wks` [{raw_data2020['Mean.temp.3wks'].dtype}] - prosječna temperatura
        - Broj nedostajućih vrijednosti: {raw_data2020['Mean.temp.3wks'].isna().sum()}
        - Raspon vrijednosti: [{raw_data2020['Mean.temp.3wks'].min():.2f} - {raw_data2020['Mean.temp.3wks'].max():.2f}]

    - `Total.precip.3wks` [{raw_data2020['Total.precip.3wks'].dtype}] - ukupne padaline [mm]
        - Broj nedostajućih vrijednosti: {raw_data2020['Total.precip.3wks'].isna().sum()}
        - Raspon vrijednosti: [{raw_data2020['Total.precip.3wks'].min():.2f} - {raw_data2020['Total.precip.3wks'].max():.2f}]

    - `Day.Average.Wind.Spd.kmph` [{raw_data2020['Day.Average.Wind.Spd.kmph'].dtype}] - prosječni dnevni vjetar
        - Broj nedostajućih vrijednosti: {raw_data2020['Day.Average.Wind.Spd.kmph'].isna().sum()}
        - Raspon vrijednosti: [{raw_data2020['Day.Average.Wind.Spd.kmph'].min():.2f} - {raw_data2020['Day.Average.Wind.Spd.kmph'].max():.2f}]

    - `Night.Average.Wind.Spd.kmph` [{raw_data2020['Night.Average.Wind.Spd.kmph'].dtype}] - prosječni noćni vjetar
        - Broj nedostajućih vrijednosti: {raw_data2020['Night.Average.Wind.Spd.kmph'].isna().sum()}
        - Raspon vrijednosti: [{raw_data2020['Night.Average.Wind.Spd.kmph'].min():.2f} - {raw_data2020['Night.Average.Wind.Spd.kmph'].max():.2f}]
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Sada kada smo se upoznali sa značenjima stupaca, **cilj ovog projekta je krenuti od najjednostavnijeg modela koji poznajemo (linearna regresija) i doći do modela kojeg su autori koristili u znanstvenom radu** kako bi istražili utjecaj vanjskih čimbenika na _prisutnost_ Varroa u promatranim kolonijama.

    Pogledajmo sada stupce i dimenzije našeg pročišćenog dataseta. Stupce smo preimenovali u kraća i Pythonu svojstvenija imena.
    """)
    return


@app.cell
def _(raw_data2020):
    column_mapping = {
        'Colony.label': 'colony_label',
        'Location': 'location',
        'Time.point': 'time_point',
        'Date': 'date',
        'Percent.mites': 'percent_mites',
        'Mean.temp.3wks': 'mean_temp',
        'Total.precip.3wks': 'total_precip',
        'Day.Average.Wind.Spd.kmph': 'day_wind',
        'Night.Average.Wind.Spd.kmph': 'night_wind'
    }

    data = raw_data2020[list(column_mapping.keys())].rename(columns=column_mapping)

    data.dropna(subset=['percent_mites'], inplace=True)
    data.head()
    return (data,)


@app.cell
def _(data, mo):
    mo.md(f"""
    Dimenzije: {data.shape}
    """)
    return


@app.cell
def _(data):
    from pandas.api.types import CategoricalDtype
    from sklearn.preprocessing import StandardScaler

    location_order = CategoricalDtype(
        categories=["British Columbia", "Southern Alberta", "Northern Alberta", "Manitoba", "Ontario", "Quebec"],
        ordered=True
    )

    data['location'] = data['location'].astype(location_order)

    data['date_d'] = data['date'].astype(float)

    scaler = StandardScaler()

    cols_to_scale = ['date_d', 'mean_temp', 'total_precip', 'day_wind', 'night_wind']

    new_colnames = ['scaled_date', 'scaled_temp', 'scaled_precip', 'scaled_day_wind', 'scaled_night_wind']

    data[new_colnames] = scaler.fit_transform(data[cols_to_scale])
    return cols_to_scale, new_colnames


@app.cell
def _(cols_to_scale, mo, new_colnames):
    mo.md(f"""
    Korištenjem `StandardScaler`-a smo skalirali stupce:

    {cols_to_scale}

    I tako dobili sljedeće stupce:

    {new_colnames}
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Stvorili smo binarnu zavisnu varijablu `varroa_is_present` koja poprima vrijednosti 0 ili 1 ovisno o tome je li uzorak zaražen.
    """)
    return


@app.cell
def _(data, np):
    data['varroa_is_present'] = np.where(data['percent_mites'] > 0, 1, 0)
    data['varroa_is_present'].value_counts()
    return


@app.cell
def _(mo):
    mo.md("""
    Krenimo od jednostavne linearne regresije:

    **varroa_is_present ~ mean_temp**
    """)
    return


@app.cell
def _(data, np, plt):
    from sklearn.linear_model import LinearRegression

    def plot_varroa_simple_regression(data):
        # Using .values converts the DF to a NumPy array to avoid name warnings
        X = data[['mean_temp']].values 
        y = data['varroa_is_present'].values

        model = LinearRegression()
        model.fit(X, y)

        x_min, x_max = X.min(), X.max()
        # Ensure X_range is 2D for the prediction
        X_range = np.linspace(x_min, x_max, 100).reshape(-1, 1)
        y_pred = model.predict(X_range)

        fig, ax = plt.subplots(figsize=(10, 6))

        # Plotting
        ax.scatter(X, y, alpha=0.4, label='Uzorci', color='steelblue')
        ax.plot(X_range, y_pred, color='firebrick', lw=2, label='OLS pravac')

        ax.set_title('varroa_is_present ~ mean_temp')
        ax.set_xlabel('Prosječna temperatura')
        ax.set_ylabel('Vjerojatnost zaraze')
        ax.legend()

        return fig, model

    simple_linear_fig, simple_linear_model = plot_varroa_simple_regression(data)
    simple_linear_fig
    return LinearRegression, simple_linear_model


@app.cell
def _(mo, simple_linear_model):
    mo.md(f"""
    Iako znamo da je nekonvencionalno koristiti linearnu regresiju u slučajevima kada imamo binarnu zavisnu varijablu, na grafu iznad vidimo da ova metoda svejedno preko OLS (_Ordinary least squares_) postiže smislen model koji modelira vjerojatnost zaraze ovisno o temperaturi.

    U nekim se područjima (poglavito ekonometrici) ovakav model stvarno i primjenjuje, primarno radi interpretabilnosti linearne regresije.

    Pogledajmo koeficijent smjera dobivenog pravca: {simple_linear_model.coef_[0]:.4f}

    Model sugerira da se za svaki porast temperature od 1°C vjerojatnost zaraze smanjuje za {abs(simple_linear_model.coef_[0]):.2%}.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Iako ima koristi od ovakvog modela, određene pretpostavke samog modela te onoga što modeliramo su prekršene.

    Možemo primijetiti da bi naš pravac u jednom trenutku prešao x-os i počeo predviđati negativne vjerojatnosti.

    Prijeđimo na malo složeniju (višestruku) linearnu regresiju:

    **varroa_is_present ~ scaled_temp + scaled_precip + scaled_night_wind**
    """)
    return


@app.cell
def _(LinearRegression, data, plt):
    def plot_multiple_regression(df):
        # Prepare features and target
        X_simple = df[['scaled_temp', 'scaled_precip', 'scaled_night_wind']].copy()
        y = df['varroa_is_present'].copy()

        # Clean data
        mask = ~(X_simple.isna().any(axis=1) | y.isna())
        X_clean = X_simple[mask]
        y_clean = y[mask]

        # Fit model
        lr_model = LinearRegression()
        lr_model.fit(X_clean, y_clean)
        y_pred_linear = lr_model.predict(X_clean)

        # Create figure and axis objects
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plotting on the 'ax' object
        ax.scatter(X_clean['scaled_temp'], y_clean, alpha=0.5, label='Uzorci')
        ax.scatter(X_clean['scaled_temp'], y_pred_linear, alpha=0.5, color='red', label='Previđanja modela')

        # Boundary lines
        ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
        ax.axhline(y=1, color='black', linestyle='--', linewidth=0.5)

        # Labels and formatting
        ax.set_xlabel('Standardizirana temperatura')
        ax.set_ylabel('Vjerojatnost zaraze')
        ax.set_title('Linearna regresija\nvarroa_is_present ~ scaled_temp + scaled_precip + scaled_night_wind')
        ax.legend()
        ax.grid(True, alpha=0.3)

        fig.tight_layout()

        return fig

    plot_multiple_regression(data)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Sada kada smo modelu dali više parametara i više "slobode", možemo lakše vidjeti kako njegova predviđanja mogu izaći iz našeg okvira vjerojatnosti [0, 1].
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Kako bismo natjerali model da previđa isključivo u rasponu (0, 1), njegov izlaz ćemo stavljati kao ulaz u sigmoidnu funkciju koja se zove logistička funkcija.

    **varroa_is_present ~ scaled_temp + scaled_precip + scaled_night_wind**

    $$\Large y_i = \beta_0 + \beta_1 x_{i1} + \beta_2 x_{i2} + \beta_3 x_{i3} + \epsilon_i \;\;\;\;\; i = 1, 2, 3,...,n$$

    $$\Large p(x) = \frac{1}{1+e^{-(\beta_0 + \beta_1 x_1 + \beta_2 x_2 + \beta_3 x_3)}}$$

    Dobiveni model je **logistička regresija** koji pripada skupini GLM-ova (_Generalized Linear Model_).
    """)
    return


@app.cell
def _(data, plt):
    from sklearn.linear_model import LogisticRegression

    def plot_logistic_predictions(df):
        X_simple = df[['scaled_temp', 'scaled_precip', 'scaled_night_wind']].copy()
        y = df['varroa_is_present'].copy()

        mask = ~(X_simple.isna().any(axis=1) | y.isna())
        X_clean = X_simple[mask]
        y_clean = y[mask]

        # 2. Modeling
        log_model = LogisticRegression(max_iter=1000, random_state=42)
        log_model.fit(X_clean, y_clean)

        y_pred_proba = log_model.predict_proba(X_clean)[:, 1]

        # 3. Plotting
        # Using subplots() is cleaner for returning figure objects
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.scatter(X_clean['scaled_temp'], y_clean, alpha=0.5, label='Uzorci')
        ax.scatter(X_clean['scaled_temp'], y_pred_proba, alpha=0.5, color='green', label='Predviđanja modela')

        ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
        ax.axhline(y=1, color='black', linestyle='--', linewidth=0.5)

        ax.set_xlabel('Standardizirana temperatura')
        ax.set_ylabel('Vjerojatnost zaraze')
        ax.set_title('Logistička regresija\nvarroa_is_present ~ scaled_temp + scaled_precip + scaled_night_wind')
        ax.legend()
        ax.grid(True, alpha=0.3)

        fig.tight_layout()

        return fig

    plot_logistic_predictions(data)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Sva predviđanja su sada u dozvoljenom rasponu.

    Sada ćemo postupno činiti model složenijim dok ne dođemo do jednog od modela kojeg su istraživači koristili.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    Kreiramo još dva modela logističke regresije tako da jednom dodajemo `scaled_date` kao parametar, a drugom uz to i `location`.

    `glm1`:

    **varroa_is_present ~ scaled_precip + scaled_temp + scaled_night_wind + scaled_date**

    `glm2`:

    **varroa_is_present ~ scaled_precip + scaled_temp + scaled_night_wind + scaled_date + location**
    """)
    return


@app.cell
def _(data):
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    # Prepare the dataset for modeling
    # Need to convert location to numeric codes for sklearn, but statsmodels can handle categorical
    glm_data = data[['varroa_is_present', 'scaled_precip', 'scaled_temp',
                            'scaled_night_wind', 'location', 'scaled_date', 'colony_label']].copy()

    formula1 = 'varroa_is_present ~ scaled_precip + scaled_temp + scaled_night_wind + scaled_date'
    glm1 = smf.glm(formula=formula1, data=glm_data, family=sm.families.Binomial()).fit()

    formula2 = 'varroa_is_present ~ scaled_precip + scaled_temp + scaled_night_wind + scaled_date + C(location)'
    glm2 = smf.glm(formula=formula2, data=glm_data, family=sm.families.Binomial()).fit()
    return glm1, glm2, glm_data, sm, smf


@app.cell
def _(mo):
    mo.md("""
    Sada uvodimo modele s interakcijskim efektima, jedan model za `scaled_night_wind`, a drugi za `scaled_day_wind`:

    `glm3`:

    **varroa_is_present ~ scaled_precip * scaled_temp * scaled_night_wind + scaled_date + location**

    `glm4`:

    **varroa_is_present ~ varroa_is_present ~ scaled_precip * scaled_temp * scaled_day_wind + scaled_date + location**

    Iz notacije možda izgleda da se prvi parametar samo sastoji od tri pomnožene značajke, ali simbolima množenja se ustvari označava da model uzima u obzir sva 3 parametra kao samostalne efekte, sve njihove kombinacije kao parove, i konačno njihov "trostrani" efekt.
    """)
    return


@app.cell
def _(data, glm_data, mo, sm, smf):
    formula3 = 'varroa_is_present ~ scaled_precip*scaled_temp*scaled_night_wind + scaled_date + C(location)'
    glm3 = smf.glm(formula=formula3, data=glm_data, family=sm.families.Binomial()).fit()

    glm_data_day_wind = data[['varroa_is_present', 'scaled_precip', 'scaled_temp',
                            'scaled_day_wind', 'location', 'scaled_date', 'colony_label']].copy()

    formula3_day_wind = 'varroa_is_present ~ scaled_precip*scaled_temp*scaled_day_wind + scaled_date + C(location)'
    glm3_day_wind = smf.glm(formula=formula3_day_wind, data=glm_data_day_wind, family=sm.families.Binomial()).fit()

    with mo.redirect_stdout():
        print(glm3.summary())
    return glm3, glm3_day_wind


@app.cell
def _(mo):
    mo.md(r"""
    Na ispisu iznad možemo vidjeti kakve sve efekte između ona 3 "pomnožena" parametra model uzima u obzir.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Iskoristimo AIC (Akaike Information Criterion) metriku kako bi usporedili koliko dobro naši GLM-ovi odgovaraju podatcima. Što je AIC niži, to je "fit" bolji.
    """)
    return


@app.cell
def _(glm1, glm2, glm3, glm3_day_wind, pd):
    comparison_df = pd.DataFrame({
        'Model': ['Simple GLM', 'GLM + Location', 'Full GLM (3-way interaction) Night Wind', 'Full GLM (3-way interaction) Day Wind'],
        'AIC': [glm1.aic, glm2.aic, glm3.aic, glm3_day_wind.aic],
    })
    comparison_df
    return


@app.cell
def _(mo):
    mo.md(r"""
    Na tablici iznad se odmah može vidjeti koliko model bolje odgovara podatcima kada uključimo lokaciju kao parametar. S druge strane, kompleksniji model sa `scaled_night_wind` interakcijskim efektima ne pokazuje nikakvo poboljšanje. Konačno, model koji uzima u obzir dnevni vjetar umjesto noćnog je najbolji među našim GLM-ovima.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Kako bi učinili zadnji korak prema modelu koji je korišten u radu, trebamo naglasiti da smo dosadašnje modele učili nad podatcima koji nisu sasvim neovisni. Prisjetimo se da je istih 240 kolonija pčela testirano za prisutnost _Varroa_ tri različita puta.

    Vizualizirajmo na koliko od ta 3 testiranja su pojedine kolonije bile zaražene.
    """)
    return


@app.cell
def _(data, plt):
    def plot_colony_presence(data):
        # 1. Calculate and sort rates
        colony_rates = data.groupby('colony_label')['varroa_is_present'].mean().sort_values()

        # 2. Create Figure and Axes objects
        # This avoids potential issues with overlapping plots in global state
        fig, ax = plt.subplots(figsize=(12, 6))

        # 3. Use the ax object for plotting
        ax.barh(range(len(colony_rates)), colony_rates.values, height=1.0, color='#1f77b4')

        # 4. Aesthetics (calling methods on 'ax' instead of 'plt')
        ax.set_title('Prisutnost Varroa po kolonijama', fontsize=14)
        ax.set_xlabel('Učestalost Varroa', fontsize=12)
        ax.set_ylabel('Kolonija', fontsize=12)

        ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_xlim(0, 1.05)
        ax.set_ylim(-5, len(colony_rates) + 5) 

        ax.grid(axis='x', linestyle='-', alpha=0.3)
        fig.tight_layout()

        return fig

    plot_colony_presence(data)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Graf iznad pokazuje da su neke kolonije bile zaražene na svakom testu, a mnoge nikada nisu bile zaražene. Ovakvu distribuciju može objasniti mnogo faktora, a kako bi naš model mogao obuhvatiti te varijacije između kolonija (koje jednostavno mogu i prirodno/genetski biti podložnije _Varroa_ parazitima od drugih) uvodimo `colony_label` kao tzv. _random effect_. Modeli koji uzimaju u obzir ove "nasumične efekte" su mješoviti (_mixed_) modeli, a naš model če stoga u konačnici postati GLMM (_Generalized Linear Mixed Model_).

    Kako bi reproducirali model iz rada za koji smo se odlučili, koristimo `rpy2` paket kako bi zvali R kod iz Pythona jer Python biblioteke zasada nemaju dobru podršku za ovakve modele.

    Konačni model je:

    **varroa_is_present ~ scaled_temp * scaled_precip * scaled_night_wind + location + scaled_date + (1|colony_label)**
    """)
    return


@app.cell
def _(data, mo):
    import rpy2.robjects as ro
    from rpy2.robjects.packages import importr
    from rpy2.robjects import pandas2ri

    lme4 = importr('lme4')
    car = importr('car')

    with (ro.default_converter + pandas2ri.converter).context():
      ro.globalenv['df'] = data  # Pass your dataframe to R

    r_script_night_wind = """
    fit <- glmer(varroa_is_present ~ scaled_temp * scaled_precip * scaled_night_wind + 
                 location + scaled_date + (1|colony_label), 
                 family = "binomial", 
                 data = df, 
                 control = glmerControl(optimizer="bobyqa", optCtrl=list(maxfun=2e9)))
    res <- Anova(fit, type=2)
    """
    ro.r(r_script_night_wind)

    anova_res_night_wind = ro.r('res')

    with mo.redirect_stdout():
        print(anova_res_night_wind)
    return (ro,)


@app.cell
def _(mo):
    mo.md(r"""
    Primijetimo da p-vrijednost za trostrani interakcijski efekt `scaled_temp:scaled_precip:scaled_night_wind` (0.26) ne ukazuje na statistički značajan efekt.

    GLMM za dnevni vjetar:

    **varroa_is_present ~ scaled_temp * scaled_precip * scaled_day_wind + location + scaled_date + (1|colony_label)**
    """)
    return


@app.cell
def _(mo, ro):
    r_script_day_wind = """
    fit <- glmer(varroa_is_present ~ scaled_temp * scaled_precip * scaled_day_wind + 
                 location + scaled_date + (1|colony_label), 
                 family = "binomial", 
                 data = df, 
                 control = glmerControl(optimizer="bobyqa", optCtrl=list(maxfun=2e9)))
    res <- Anova(fit, type=2)
    """
    ro.r(r_script_day_wind)

    anova_res_day_wind = ro.r('res')

    with mo.redirect_stdout():
        print(anova_res_day_wind)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Ovdje ipak možemo primijetiti da p-vrijednost za trostrani interakcijski efekt `scaled_temp:scaled_precip:scaled_day_wind` (0.01) _ukazuje_ na statistički značajan efekt. Po čistoj intuiciji, razlika u efektu dnevnog i noćnog vjetra ima smisla jer pčele lete danju i stoga vjetar u tom periodu ima veći utjecaj na njihov let i ponašanje.
    """)
    return


if __name__ == "__main__":
    app.run()
