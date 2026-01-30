import os
import sys
import json
import random
import pandas as pd
import yaml
import copy
import numpy as np
import pickle
import duckdb
from sklearn.cluster import KMeans

# Ensure local modules can be imported
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

from world.definitions import Ad, User

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

def calculate_similarity(user1, user2):
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
        0.03 * age_similarity + 0.02 * family_similarity + 0.005 * gender_similarity +
        0.08 * hobby_similarity + 0.04 * profession_similarity + 0.05 * activity_similarity +
        0.04 * risk_similarity + 0.03 * social_similarity + 0.05 * friend_of_friend
    )
    return compatibility

def friendship_simulation(users, friendship_threshold):
    for i in range(len(users)):
        for j in range(i + 1, len(users)):
            user1, user2 = users[i], users[j]
            compatibility = calculate_similarity(user1, user2)
            p_friendship = 0.7 * compatibility + 0.3 * np.random.uniform(0, 1)
            if p_friendship > friendship_threshold:
                user1.add_friend(user2.user_id)
                user2.add_friend(user1.user_id)

def init():
    print("Loading config...")
    with open("context/simulation_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    print("Generating ads...")
    ads_features_df = pd.read_csv("data/ads_features.csv")
    all_ads = []
    for _, row in ads_features_df.iterrows():
        row_dict = row.to_dict()
        row_dict['description'] = generate_ad_description(row_dict)
        ad = Ad(**row_dict)
        all_ads.append(ad)

    schedule = day_of_entry_assignment(all_ads, config)
    for ad in all_ads:
        for day, ad_ids in schedule.items():
            if ad.ad_id in ad_ids:
                ad.day_of_entry = day
                break

    with open("data/all_ads.pkl", "wb") as f:
        pickle.dump(all_ads, f)
    print(f"Saved {len(all_ads)} ads to pkl.")

    print("Generating users...")
    users_df = pd.read_csv("data/users_features.csv")
    all_users = [User(**row.to_dict()) for _, row in users_df.iterrows()]
    friendship_simulation(all_users, config["friendship_threshold"])
    with open("data/all_users.pkl", "wb") as f:
        pickle.dump(all_users, f)
    print(f"Saved {len(all_users)} users to pkl.")

    print("Setting up database...")
    db_path = "data/databases/simulation_db.duckdb"
    if os.path.exists(db_path):
        os.remove(db_path)
    con = duckdb.connect(db_path)
    
    con.execute("""
    CREATE TABLE ads (
        ad_id BIGINT PRIMARY KEY, "group" VARCHAR, emotion_label VARCHAR, message_type VARCHAR,
        visual_style VARCHAR, num_people INTEGER, people_present BOOLEAN, people_area_ratio FLOAT,
        product_present BOOLEAN, product_area_ratio FLOAT, object_count INTEGER, object_list JSON,
        dominant_element VARCHAR, text_present BOOLEAN, text_area_ratio FLOAT, avg_font_size_proxy FLOAT,
        dominant_colors JSON, brightness_category VARCHAR, saturation_category VARCHAR, hue_category VARCHAR,
        visual_impact FLOAT, description TEXT, is_active BOOLEAN, day_of_entry INTEGER
    );
    """)
    
    for ad in all_ads:
        con.execute("INSERT INTO ads VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", [
            ad.ad_id, ad.group, ad.emotion_label, ad.message_type, ad.visual_style, ad.num_people,
            ad.people_present, ad.people_area_ratio, ad.product_present, ad.product_area_ratio,
            ad.object_count, json.dumps(ad.object_list), ad.dominant_element, ad.text_present,
            ad.text_area_ratio, ad.avg_font_size_proxy, json.dumps(ad.dominant_colors),
            ad.brightness_category, ad.saturation_category, ad.hue_category, ad.visual_impact,
            ad.description, ad.is_active, ad.day_of_entry
        ])

    con.execute("""
    CREATE TABLE users (
        user_id BIGINT PRIMARY KEY, gender VARCHAR, age INTEGER, profession JSON,
        hobby JSON, family VARCHAR, friend_list JSON
    );
    """)
    for user in all_users:
        con.execute("INSERT INTO users VALUES (?,?,?,?,?,?,?)", [
            user.user_id, user.gender, user.age, json.dumps(user.profession),
            json.dumps(user.hobby), user.family, json.dumps(user.friend_list)
        ])

    con.execute("""
    CREATE TABLE interactions (
        interaction_id BIGINT PRIMARY KEY, ad_id BIGINT, interaction_rate FLOAT, user_id BIGINT,
        acute_irritation INTEGER, acute_interest INTEGER, acute_arousal INTEGER, 
        bias_irritation INTEGER, bias_trust INTEGER, bias_fatigue INTEGER,
        acute_irritation_change INTEGER, acute_interest_change INTEGER, acute_arousal_change INTEGER,
        bias_irritation_change INTEGER, bias_trust_change INTEGER, bias_fatigue_change INTEGER,
        "ignore" BOOLEAN, click BOOLEAN, "like" BOOLEAN, dislike BOOLEAN, share INTEGER,
        reaction_description TEXT, prompt TEXT, timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)
    con.execute("CREATE SEQUENCE interaction_id_seq;")

    con.execute("CREATE TABLE users_grouping (grouping_id BIGINT PRIMARY KEY, user_id BIGINT, day INTEGER, \"group\" INTEGER, model TEXT);")
    con.execute("CREATE SEQUENCE grouping_id_seq;")
    
    # Simple KMeans grouping
    users_data = []
    for user in all_users:
        users_data.append({
            "user_id": user.user_id,
            "gender": user.gender,
            "age": user.age,
            "profession": tuple(sorted(eval(user.profession))),
            "hobby": tuple(sorted(eval(user.hobby))),
            "family": user.family,
            "activity_level": user.activity_level,
            "risk_tolerance": user.risk_tolerance,
            "social_engagement": user.social_engagement,
        })
    df = pd.DataFrame(users_data)
    df_encoded = pd.get_dummies(df, columns=["gender", "profession", "hobby", "family"])
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    df_encoded["group"] = kmeans.fit_predict(df_encoded.drop(columns=["user_id"]))
    
    for _, row in df_encoded.iterrows():
        con.execute("INSERT INTO users_grouping VALUES (nextval('grouping_id_seq'), ?, 0, ?, 'KMeans_initial')", [
            row["user_id"], row["group"]
        ])
    
    print("Database initialization complete.")
    con.close()

if __name__ == "__main__":
    init()
