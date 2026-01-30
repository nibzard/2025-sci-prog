import marimo

__generated_with = "0.19.4"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import duckdb
    import json
    import random
    import pandas as pd
    import yaml
    import copy
    import numpy as np
    import pickle
    import os
    import sys
    
    if os.getcwd() not in sys.path:
        sys.path.append(os.getcwd())
        
    from IPython.display import display, HTML
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import OneHotEncoder
    from world.definitions import Ad, User
    return Ad, HTML, KMeans, OneHotEncoder, User, copy, display, duckdb, json, mo, np, os, pd, pickle, random, sys, yaml


@app.cell
def _(mo):
    mo.md(r"""
    # World Definition & Ad Creation
    """)
    return


@app.cell
def _(copy, yaml):
    with open("context/simulation_config.yaml", "r") as config_file:
        config_str = config_file.read()

    _loaded_config = yaml.safe_load(config_str)

    config = copy.deepcopy(_loaded_config)
    return (config,)


@app.cell
def _(config):
    config
    return


@app.cell
def _(yaml):
    STATE_FILE = "context/simulation_state.yaml"

    def load_state():
        with open(STATE_FILE, "r") as state_read_file:
            return yaml.safe_load(state_read_file)

    def save_state(state):
        with open(STATE_FILE, "w") as state_write_file:
            yaml.safe_dump(state, state_write_file)

    state = load_state()
    return (state,)


@app.cell
def _(state):
    state
    return


@app.cell
def _(pd):
    ads_features_df = pd.read_csv("data/ads_features.csv")
    ads_features_df
    return (ads_features_df,)


@app.cell
def _(random):
    def day_of_entry_assignment(all_ads, config):
        days_ads_can_enter = config["days_ads_can_enter"]
        new_ads_per_day = config["new_ads_per_day"]

        ad_ids = [ad.ad_id for ad in all_ads]
        random.shuffle(ad_ids)

        schedule = {day: [] for day in days_ads_can_enter}
        day_index = 0

        for ad_id in ad_ids:
            if day_index >= len(days_ads_can_enter):
                schedule[days_ads_can_enter[0]].append(ad_id)
                continue

            day = days_ads_can_enter[day_index]

            if len(schedule[day]) < new_ads_per_day:
                schedule[day].append(ad_id)
            else:
                day_index += 1
                if day_index < len(days_ads_can_enter):
                    next_day = days_ads_can_enter[day_index]
                    schedule[next_day].append(ad_id)
                else:
                    schedule[days_ads_can_enter[0]].append(ad_id)

        return schedule

    def generate_ad_description(row):
        """Generates a friendly, human-readable objective description of an ad."""
        
        def hex_to_name(hex_code):
            hex_code = hex_code.lstrip('#')
            try:
                r, g, b = tuple(int(hex_code[i:i+2], 16) for i in (0, 2, 4))
            except:
                return "unknown color"
            
            colors = {
                'red': (255, 0, 0), 'green': (0, 255, 0), 'blue': (0, 0, 255),
                'white': (255, 255, 255), 'black': (0, 0, 0), 'yellow': (255, 255, 0),
                'cyan': (0, 255, 255), 'magenta': (255, 0, 255), 'orange': (255, 165, 0),
                'purple': (128, 0, 128), 'brown': (165, 42, 42), 'gray': (128, 128, 128),
                'pink': (255, 192, 203), 'gold': (255, 215, 0), 'beige': (245, 245, 220),
                'navy': (0, 0, 128), 'teal': (0, 128, 128)
            }
            def color_dist(c1, c2):
                return sum((a - b) ** 2 for a, b in zip(c1, c2))
            return min(colors.keys(), key=lambda k: color_dist(colors[k], (r, g, b)))

        group = row.get('group', 'unspecified').replace('_', ' ')
        emotion = row.get('emotion_label', 'neutral').replace('_', ' ')
        msg_type = row.get('message_type', 'generic').replace('_', ' ')
        style = row.get('visual_style', 'standard').replace('_', ' ')
        people = int(row.get('num_people', 0))
        product_present = row.get('product_present', False)
        
        raw_colors = row.get('dominant_colors', [])
        if isinstance(raw_colors, str):
            try:
                raw_colors = json.loads(raw_colors.replace("'", '"'))
            except:
                raw_colors = []
        
        color_names = [hex_to_name(c) for c in raw_colors[:3]]
        if color_names:
            color_str = f"The primary colors in this ad are {', '.join(color_names[:-1])} and {color_names[-1]}" if len(color_names) > 1 else f"The primary color in this ad is {color_names[0]}"
        else:
            color_str = "The specific color palette is not defined"
            
        brightness = row.get('brightness_category', 'normal')
        saturation = row.get('saturation_category', 'standard')
        impact = float(row.get('visual_impact', 0))
        text_present = row.get('text_present', False)
        
        if impact < 0.3:
            impact_desc = "is not vivid or visually appealing, but rather subtle, soft and easy-going"
        elif impact < 0.6:
            impact_desc = "is visibly balanced and clear, offering a standard level of vividness"
        else:
            impact_desc = "is exceptionally vivid and visually striking, with a powerful and clear presence"
        
        desc = f"This is an objective description of an advertisement from the {group} category. "
        desc += f"It is presented in a {style} style, utilizing {msg_type} messaging with an underlying {emotion} tone. "
        
        if people > 1:
            desc += f"The image features {people} human figures. "
        elif people == 1:
            desc += "There is one human figure present in the scene. "
        else:
            desc += "There are no human figures in this composition. "
            
        desc += f"The product is {'clearly visible' if product_present else 'not prominently featured'}, and there is {'also some text included' if text_present else 'no text present'}. "
        desc += f"{color_str}, set within a {brightness} and {saturation} atmosphere. "
        desc += f"The overall visual impact of this ad {impact_desc}."
        
        return desc

    return (day_of_entry_assignment, generate_ad_description)


@app.cell
def _(Ad, ads_features_df, config, day_of_entry_assignment, pickle):
    all_ads = []

    for _, row in ads_features_df.iterrows():
        row_dict = row.to_dict()
        row_dict['description'] = generate_ad_description(row_dict)
        ad = Ad(**row_dict)
        all_ads.append(ad)

    ads_entering_schedule = day_of_entry_assignment(all_ads, config)

    for ad in all_ads:
        for day, ad_ids in ads_entering_schedule.items():
            if ad.ad_id in ad_ids:
                ad.day_of_entry = day
                break
    
    with open("data/all_ads.pkl", "wb") as ads_pkl:
        pickle.dump(all_ads, ads_pkl)

    print("len(all_ads)")
    print(len(all_ads))
    return ads_entering_schedule, all_ads


@app.cell
def _(all_ads):
    all_ads[0].to_message_format()
    return


@app.cell
def _(config, random):
    def schedule_for_day(current_day, ads_entering_schedule, all_ads):
        if current_day in ads_entering_schedule:
            for ad_id in ads_entering_schedule[current_day]:
                for ad in all_ads:
                    if ad.ad_id == ad_id:
                        ad.is_active = True
                        ad.day_of_entry = current_day

        active_ads = [ad for ad in all_ads if ad.is_active]

        if len(active_ads) > config["max_ads_shown_per_day"]:
            return random.sample(active_ads, config["max_ads_shown_per_day"])

        return active_ads
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Agent Template Creation
    """)
    return


@app.cell
def _(np):
    def calculate_similarity(user1, user2, users):
        age_similarity = np.exp(-abs(user1.age - user2.age) / 15)
        family_similarity = 1 if user1.family == user2.family else 0
        gender_similarity = 1 if user1.gender == user2.gender else 0

        hobbies1 = set(eval(user1.hobby))
        hobbies2 = set(eval(user2.hobby))
        hobby_similarity = len(hobbies1.intersection(hobbies2)) / len(hobbies1.union(hobbies2)) if len(hobbies1.union(hobbies2)) > 0 else 0

        professions1 = set(eval(user1.profession))
        professions2 = set(eval(user2.profession))
        profession_similarity = 1 if professions1 == professions2 else 0

        activity_similarity = 1 - abs(user1.activity_level - user2.activity_level) / 100
        risk_similarity = 1 - abs(user1.risk_tolerance - user2.risk_tolerance) / 100
        social_similarity = 1 - abs(user1.social_engagement - user2.social_engagement) / 100

        mutual_friends = len(set(user1.friend_list).intersection(set(user2.friend_list)))
        friend_of_friend = 0.1 * mutual_friends

        compatibility = (
            0.03 * age_similarity
            + 0.02 * family_similarity
            + 0.005 * gender_similarity
            + 0.08 * hobby_similarity
            + 0.04 * profession_similarity
            + 0.05 * activity_similarity
            + 0.04 * risk_similarity
            + 0.03 * social_similarity
            + 0.05 * friend_of_friend
        )

        return compatibility
    return (calculate_similarity,)


@app.cell
def _(calculate_similarity, np):
    def friendship_simulation(users, friendship_threshold):
        for i in range(len(users)):
            for j in range(i + 1, len(users)):
                user1 = users[i]
                user2 = users[j]

                compatibility = calculate_similarity(user1, user2, users)
                random_noise = np.random.uniform(0, 1)

                p_friendship = 0.7 * compatibility + 0.3 * random_noise

                if p_friendship > friendship_threshold:
                    user1.add_friend(user2.user_id)
                    user2.add_friend(user1.user_id)
    return (friendship_simulation,)


@app.cell
def _(User, config, friendship_simulation, pd, pickle):
    users_df = pd.read_csv("data/users_features.csv")
    all_users = [User(**row.to_dict()) for _, row in users_df.iterrows()]
    friendship_simulation(all_users, config["friendship_threshold"])
    
    with open("data/all_users.pkl", "wb") as users_pkl:
        pickle.dump(all_users, users_pkl)
        
    return all_users, users_df


@app.cell
def _(all_users):
    for u in all_users:
        print(u.friend_list)
        print(len(u.friend_list))
    return


@app.cell
def _(all_users, config, engine, mo):
    mo.md(r"""
    # Persona Refactoring
    """)
    
    unique_professions = set()
    for user in all_users:
        if user.profession:
             # profession is stored as string representation of list "['writer']"
            prof_list = eval(user.profession)
            if prof_list:
                unique_professions.add(prof_list[0])
                
    print(f"Refactoring personas for {len(unique_professions)} professions...")
    
    # Simple cache to avoid re-generating in same session
    refactored_cache = {}
    
    for user in all_users:
        prof_list = eval(user.profession)
        if not prof_list: continue
        job = prof_list[0]
        
        if job in refactored_cache:
            user.persona_narrative = refactored_cache[job]
            continue
            
        if user.persona_narrative:
            # Refactor existing persona
            prompt = f"""
            You are a professional editor and creative writer.
            I will provide you with a raw text describing a {job}.
            
            RAW TEXT:
            {user.persona_narrative}
            
            YOUR TASK:
            Refactor this text to be a rich, authenticated first-person narrative.
            
            GUIDELINES:
            1. FIRST PERSON: Change all "he/she" to "I". The user IS this person.
            2. PERSONALITY: If the text is dry or corporate, add personality. Mention specific frustrations, joys, and daily rhythms.
            3. PERSONAL INFO CHECK: Read the text. Does it contain actual personal details (hobbies, family, routine)? 
               - If YES: Preserve them perfectly.
               - If NO (or if it just talks about the company): Invent plausible, specific personal details that fit the profession.
            4. LENGTH: Keep it between 150-300 words.
            5. TONE: Human, candid, slightly informal but professional.
            
            Output ONLY the refined text.
            """
            
            try:
                # Use query_llm from engine
                # Note: This might take time for all professions
                print(f"Refactoring {job}...")
                new_narrative = engine.query_llm(prompt, config, json_mode=False)
                if new_narrative:
                    user.persona_narrative = new_narrative
                    refactored_cache[job] = new_narrative
            except Exception as e:
                print(f"Error refactoring {job}: {e}")
                
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Interaction History Management System
    """)
    return


@app.cell
def _(duckdb):
    def create_tables(db_path):
        con = duckdb.connect(db_path)

        con.execute("""
        CREATE TABLE IF NOT EXISTS ads (
            ad_id BIGINT NOT NULL,
            "group" VARCHAR(64) NOT NULL,
            emotion_label VARCHAR(64) NOT NULL,
            message_type VARCHAR(64) NOT NULL,
            visual_style VARCHAR(64) NOT NULL,
            num_people INTEGER NOT NULL,
            people_present BOOLEAN NOT NULL,
            people_area_ratio FLOAT NOT NULL,
            product_present BOOLEAN NOT NULL,
            product_area_ratio FLOAT NOT NULL,
            object_count INTEGER NOT NULL,
            object_list JSON,
            dominant_element VARCHAR(64),
            text_present BOOLEAN NOT NULL,
            text_area_ratio FLOAT NOT NULL,
            avg_font_size_proxy FLOAT NOT NULL,
            dominant_colors JSON,
            brightness_category VARCHAR(32) NOT NULL,
            saturation_category VARCHAR(32) NOT NULL,
            hue_category VARCHAR(32) NOT NULL,
            visual_impact FLOAT NOT NULL,
            description TEXT,
            is_active BOOLEAN NOT NULL,
            day_of_entry INTEGER NOT NULL,
            PRIMARY KEY (ad_id)
        );
        """)

        con.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id BIGINT NOT NULL,
            gender VARCHAR(32) NOT NULL,
            age INTEGER NOT NULL,
            profession JSON,
            hobby JSON,
            family VARCHAR(64) NOT NULL,
            friend_list JSON,
            PRIMARY KEY (user_id)
        );
        """)

        con.execute("""
        CREATE TABLE IF NOT EXISTS interactions (
            interaction_id BIGINT NOT NULL,
            ad_id BIGINT NOT NULL,
            interaction_rate FLOAT NOT NULL,
            user_id BIGINT NOT NULL,
            acute_irritation INTEGER NOT NULL,
            acute_interest INTEGER NOT NULL,
            acute_arousal INTEGER NOT NULL,
            bias_irritation INTEGER NOT NULL,
            bias_trust INTEGER NOT NULL,
            bias_fatigue INTEGER NOT NULL,
            acute_irritation_change INTEGER NOT NULL,
            acute_interest_change INTEGER NOT NULL,
            acute_arousal_change INTEGER NOT NULL,
            bias_irritation_change INTEGER NOT NULL,
            bias_trust_change INTEGER NOT NULL,
            bias_fatigue_change INTEGER NOT NULL,
            "ignore" BOOLEAN NOT NULL,
            click BOOLEAN NOT NULL,
            "like" BOOLEAN NOT NULL,
            dislike BOOLEAN NOT NULL,
            share INTEGER NOT NULL,
            reaction_description TEXT NOT NULL,
            prompt TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (interaction_id),
            FOREIGN KEY (ad_id) REFERENCES ads(ad_id),
            FOREIGN KEY (user_id) REFERENCES users(user_id)
        );
        """)

        con.execute("""
        CREATE SEQUENCE IF NOT EXISTS grouping_id_seq;
        CREATE TABLE IF NOT EXISTS users_grouping (
            grouping_id BIGINT NOT NULL DEFAULT nextval('grouping_id_seq'),
            user_id BIGINT NOT NULL,
            day INTEGER NOT NULL,
            "group" INTEGER NOT NULL,
            model TEXT,
            confidence INTEGER,
            PRIMARY KEY (grouping_id),
            FOREIGN KEY (user_id) REFERENCES users(user_id)
        );
        """)

    create_tables("data/databases/simulation_db.duckdb")
    
    # Ensure persona column exists (migration for existing DBs)
    con = duckdb.connect("data/databases/simulation_db.duckdb")
    try:
        con.execute("ALTER TABLE users ADD COLUMN persona TEXT")
        print("Added persona column to users table.")
    except:
        # Column likely already exists
        pass
    con.close()
    return


@app.cell
def _(all_ads, all_users, duckdb, json):
    def insert_ads_and_users(db_path, all_ads, all_users):
        con = duckdb.connect(db_path)

        for ad in all_ads:
            con.execute("""
                INSERT INTO ads (
                    ad_id, "group", emotion_label, message_type, visual_style,
                    num_people, people_present, people_area_ratio,
                    product_present, product_area_ratio,
                    object_count, object_list, dominant_element,
                    text_present, text_area_ratio, avg_font_size_proxy,
                    dominant_colors, brightness_category, saturation_category,
                    hue_category, visual_impact, is_active, day_of_entry
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                ad.ad_id,                              
                ad.group,
                ad.emotion_label,
                ad.message_type,
                ad.visual_style,
                ad.num_people,
                ad.people_present,
                ad.people_area_ratio,
                ad.product_present,
                ad.product_area_ratio,
                ad.object_count,
                json.dumps(ad.object_list),
                ad.dominant_element,
                ad.text_present,
                ad.text_area_ratio,
                ad.avg_font_size_proxy,
                json.dumps(ad.dominant_colors),
                ad.brightness_category,
                ad.saturation_category,
                ad.hue_category,
                ad.visual_impact,
                ad.description,
                ad.is_active,
                ad.day_of_entry,
            ])

        for user in all_users:
            con.execute("""
                INSERT INTO users (
                    user_id, gender, age, profession, hobby, family, friend_list, persona
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                user.user_id,
                user.gender,
                user.age,
                json.dumps(user.profession),
                json.dumps(user.hobby),
                user.family,
                json.dumps(user.friend_list),
                user.persona_narrative
            ])

        con.close()

    insert_ads_and_users(
        db_path="data/databases/simulation_db.duckdb",
        all_ads=all_ads,
        all_users=all_users
    )
    return


@app.cell
def _(duckdb):
    def inspect_duckdb(db_path):
        con = duckdb.connect(db_path)

        tables = con.execute("SHOW TABLES").fetchall()
        if not tables:
            print("Database is empty.")
            return

        print(f"Found {len(tables)} table(s): {[t[0] for t in tables]}")

        for table, in tables:
            print(f"\n=== Table: {table} ===")

            columns = con.execute(f"DESCRIBE {table}").fetchall()
            print("Columns (name | type):")
            for col in columns:
                col_name = col[0]
                col_type = col[1]
                print(f"  {col_name} | {col_type}")

            count = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            print(f"Total rows: {count}")

        con.close()

    inspect_duckdb("data/databases/simulation_db.duckdb")
    return


@app.cell
def _(HTML, display, duckdb):
    def display_table(db_path: str, table_name: str, max_height_px=400):
        con = duckdb.connect(db_path)

        df = con.execute(f"SELECT * FROM {table_name}").df()
        con.close()

        html = df.to_html(index=False, escape=False)

        display(HTML(f"""
            <div style="
                max-height: {max_height_px}px;
                overflow-y: auto;
                overflow-x: auto;
                white-space: nowrap;
                border: 1px solid #ccc;
            ">
                {html}
            </div>
        """))
    return (display_table,)


@app.cell
def _(display_table):
    display_table(
        "data/databases/simulation_db.duckdb",
        "users"
    )
    return


@app.cell
def _(display_table):
    display_table(
        "data/databases/simulation_db.duckdb",
        "ads"
    )
    return


@app.cell
def _(display_table):
    display_table(
        "data/databases/simulation_db.duckdb",
        "interactions"
    )
    return


@app.cell
def _(display_table):
    display_table(
        "data/databases/simulation_db.duckdb",
        "users_grouping"
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Grouping model
    """)
    return


@app.cell
def _(all_users, duckdb, pd):
    def grouping_model(all_users, db_path="data/databases/simulation_db.duckdb"):
        users_data = []
        for user in all_users:
            users_data.append(
                {
                    "user_id": user.user_id,
                    "gender": user.gender,
                    "age": user.age,
                    "profession": tuple(sorted(eval(user.profession))),
                    "hobby": tuple(sorted(eval(user.hobby))),
                    "family": user.family,
                    "activity_level": user.activity_level,
                    "risk_tolerance": user.risk_tolerance,
                    "social_engagement": user.social_engagement,
                    "friend_list_len": len(user.friend_list),
                }
            )
        df = pd.DataFrame(users_data)

        df_encoded = pd.get_dummies(
            df, columns=["gender", "profession", "hobby", "family"]
        )

        from sklearn.cluster import KMeans

        kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
        df_encoded["group"] = kmeans.fit_predict(df_encoded.drop(columns=["user_id"]))

        con = duckdb.connect(db_path)
        for _, row in df_encoded.iterrows():
            con.execute("""
                INSERT INTO users_grouping (
                    grouping_id, user_id, day, "group", model
                )
                VALUES (nextval('grouping_id_seq'), ?, ?, ?, ?)
            """, [
                row["user_id"],
                0,
                row["group"],
                "KMeans_initial",
            ])

    grouping_df = grouping_model(all_users)
    grouping_df
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

