import streamlit as st
import pandas as pd
import json
import time
import sys
import os

# Ensure the root directory is in sys.path
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_path not in sys.path:
    sys.path.append(root_path)

from world import engine

# Page setup
st.set_page_config(page_title="AdSim Interactive Visualizer", layout="wide")

# Initialization
engine.load_env()
config = engine.load_config()
all_ads, all_users = engine.load_data()
user_groups_df = engine.fetch_groupings()

if "sim_state" not in st.session_state:
    st.session_state.sim_state = engine.load_simulation_state()

st.title("🚀 Ad Simulation Interactive Visualizer")
st.markdown("---")

# --- FRAME 1: Configuration & Control ---
st.header("⚙️ Frame 1: System Configuration")
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    backend = st.selectbox("LLM Backend", ["openai", "google", "local"], index=0)
    # Dynamic model fetching
    available_models = engine.fetch_available_models(backend)
    selected_model = st.selectbox("Select Model", available_models)

with col2:
    curr_day = st.session_state.sim_state.get("current_simulation_day", 0)
    sched_ads = st.session_state.sim_state.get("ads_scheduled_for_day", [])
    st.write(f"**Current Day:** {curr_day}")
    st.write(f"**Ads in Queue:** {len(sched_ads)}")
    if sched_ads:
        next_ad_id = sched_ads[0]
        st.write(f"**Next Ad to Process:** {next_ad_id}")
        
        # Show Ad Description for Context
        next_ad = next((ad for ad in all_ads if ad.ad_id == next_ad_id), None)
        if next_ad and hasattr(next_ad, 'description'):
            st.info(f"📢 **Ad Context:** {next_ad.description}")
            
    else:
        st.write("✨ Queue empty. Click start to schedule new ads.")

with col3:
    start_step = st.button("▶️ Start Simulation Step", use_container_width=True)

# Prompt Customization
with st.expander("📝 Edit Base Prompt Template", expanded=False):
    st.markdown("Customize how the agents see the ad. Use `{user_features}` and `{ad_json}` as placeholders.")
    custom_prompt = st.text_area(
        "Prompt Template",
        value=engine.DEFAULT_PROMPT_TEMPLATE,
        height=400,
        help="This template is used for every agent interaction."
    )

st.markdown("---")

# --- FRAME 2: Live Progress & Decision Visualization ---
st.header("🧠 Frame 2: Live Progress & Decisions")
progress_container = st.container()

if start_step:
    state = st.session_state.sim_state
    
    if not sched_ads:
        with progress_container:
            # Increment day BEFORE scheduling, as requested
            state["current_simulation_day"] += 1
            curr_day = state["current_simulation_day"]
            
            st.info(f"📅 Scheduling ads for Day {curr_day}...")
            sched_ads = engine.schedule_for_day(curr_day, all_ads, config)
            state["ads_scheduled_for_day"] = sched_ads
            engine.save_simulation_state(state)
            st.success(f"Scheduled {len(sched_ads)} ads for Day {curr_day}.")
            st.rerun()
    else:
        ad_id = sched_ads[0] # Peek at current ad
        target_ad = next((ad for ad in all_ads if ad.ad_id == ad_id), None)
        
        if target_ad:
            with progress_container:
                st.subheader(f"🎯 Ad {ad_id}: Deep-Dive Processing")
                
                # 1. Selection Logic
                st.markdown("### 🔍 Step 1: Targeted Selection")
                user_groups_df = engine.fetch_groupings()
                
                selected_agents, selection_stats = engine.select_agents(
                    target_ad, all_users, config, user_groups_df, return_stats=True
                )
                
                st.write("**Selection Details per Group:**")
                stats_df = pd.DataFrame(selection_stats)
                # Convert list of IDs to string for better display
                stats_df['selected_ids'] = stats_df['selected_ids'].apply(lambda x: ", ".join(map(str, x)))
                st.dataframe(stats_df, use_container_width=True)
                st.info(f"Total Selected for Interaction: **{len(selected_agents)}** agents.")

                # 2. Detailed Interactions
                st.markdown("### 🧠 Step 2: Individual Interactions")
                interactions_results = []
                bar = st.progress(0)
                
                for i, agent in enumerate(selected_agents):
                    with st.status(f"Processing User {agent.user_id}...", expanded=(i==0)) as status:
                        prompt = engine.format_prompt(agent, target_ad, custom_template=custom_prompt)
                        response_str = engine.query_llm(prompt, config, backend, selected_model)
                        
                        try:
                            if not response_str or response_str == "{}":
                                raise ValueError("Empty response")
                                
                            response_json = json.loads(response_str)
                            
                            # Log Initial State for Comparison
                            old_state = agent.emotional_state.copy()
                            
                            # Process
                            engine.update_user_emotional_state(agent, response_json)
                            
                            interaction = {
                                "user_id": agent.user_id,
                                "ad_id": target_ad.ad_id,
                                "interaction_rate": target_ad.interaction_rate,
                                "acute_irritation": agent.emotional_state["acute_irritation"],
                                "acute_interest": agent.emotional_state["acute_interest"],
                                "acute_arousal": agent.emotional_state["acute_arousal"],
                                "bias_irritation": agent.emotional_state["bias_irritation"],
                                "bias_trust": agent.emotional_state["bias_trust"],
                                "bias_fatigue": agent.emotional_state["bias_fatigue"],
                                "acute_irritation_change": response_json.get("acute_irritation_change", 0),
                                "acute_interest_change": response_json.get("acute_interest_change", 0),
                                "acute_arousal_change": response_json.get("acute_arousal_change", 0),
                                "bias_irritation_change": response_json.get("bias_irritation_change", 0),
                                "bias_trust_change": response_json.get("bias_trust_change", 0),
                                "bias_fatigue_change": response_json.get("bias_fatigue_change", 0),
                                "ignore": response_json.get("ignore", False),
                                "click": response_json.get("click", False),
                                "like": response_json.get("like", False),
                                "dislike": response_json.get("dislike", False),
                                "share": response_json.get("share", 0),
                                "reaction_description": response_json.get("reaction_description", ""),
                                "prompt": prompt
                            }
                            
                            engine.save_interaction_to_db(interaction)
                            interactions_results.append(interaction)
                            
                            # Display Rich Narrative Details
                            c_persona, c_context, c_react = st.columns(3)
                            
                            with c_persona:
                                st.markdown("#### 👤 Persona")
                                if hasattr(agent, 'persona_narrative') and agent.persona_narrative:
                                    st.caption(agent.persona_narrative)
                                else:
                                    st.caption("_No narrative persona loaded._")
                                    
                            with c_context:
                                st.markdown("#### 📅 Today's Context")
                                daily_events = getattr(agent, "daily_event_summaries", [])
                                if daily_events:
                                    for ev in daily_events:
                                        st.text(f"• {ev}")
                                else:
                                    st.caption("_No significant events today._")

                            with c_react:
                                st.markdown("#### ⚡ Reaction")
                                if interaction['reaction_description']:
                                    st.write(f"_{interaction['reaction_description']}_")
                                else:
                                    st.caption("_No internal monologue._")
                                    
                                # Action Badges
                                badges = []
                                if interaction['click']: badges.append("✅ Clicked")
                                if interaction['like']: badges.append("❤️ Liked")
                                if interaction['share']: badges.append("📢 Shared")
                                if interaction['dislike']: badges.append("👎 Disliked")
                                if interaction['ignore']: badges.append("⏭️ Ignored")
                                
                                st.markdown(" ".join([f"`{b}`" for b in badges]))

                            if interaction['share'] > 0:
                                friend_id = engine.handle_share_action(agent, target_ad, interaction['share'])
                                if friend_id:
                                    st.success(f"📣 **Ad Propagated!** Shared with User {friend_id}")
                            
                            # Show Emotional State Change
                            st.write("**Emotional State Shift:**")
                            cols = st.columns(6)
                            for idx, (key, val) in enumerate(agent.emotional_state.items()):
                                change = val - old_state[key]
                                color = "normal" if change == 0 else "inverse" if change < 0 else "normal"
                                cols[idx % 6].metric(key.replace("bias_", "B: ").replace("acute_", "A: "), 
                                                    f"{val}", delta=f"{change:+}")

                            # Update Ad Rate
                            target_ad.update_interaction_rate(
                                click=1 if interaction['click'] else 0,
                                share=1 if interaction['share'] > 0 else 0,
                                like=1 if interaction['like'] else 0,
                                dislike=1 if interaction['dislike'] else 0,
                                ignore=1 if interaction['ignore'] else 0
                            )
                            status.update(label=f"User {agent.user_id} interaction complete!", state="complete")

                        except Exception as e:
                            st.error(f"Error for User {agent.user_id}: {e}")
                            status.update(label=f"User {agent.user_id} failed!", state="error")
                    
                    bar.progress((i + 1) / len(selected_agents))

                # After ad processing
                st.session_state.last_results = interactions_results
                sched_ads.pop(0)
                state["ads_scheduled_for_day"] = sched_ads
                engine.save_simulation_state(state)
                st.success(f"Simulation step for Ad {ad_id} finalized.")

# --- FRAME 3: Final Results & Loop Control ---
st.markdown("---")
st.header("📊 Frame 3: Results Summary")

if "last_results" in st.session_state and st.session_state.last_results:
    results = st.session_state.last_results
    df = pd.DataFrame(results)
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Users Reached", len(df))
    c2.metric("Total Clicks", df['click'].sum())
    c3.metric("Total Likes", df['like'].sum())
    c4.metric("Total Shares", df['share'].sum())
    
    st.subheader("Interaction Details")
    st.dataframe(df[['user_id', 'click', 'like', 'share', 'reaction_description']])
    
    if st.button("🔁 Process Another Loop (Next Ad/Day)"):
        st.rerun()
else:
    st.write("_Run a simulation step to see results here._")
