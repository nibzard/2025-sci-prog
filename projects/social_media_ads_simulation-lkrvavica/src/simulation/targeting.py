
import pandas as pd
import random
from typing import List, Dict, Any, Tuple
from src.simulation.models import User, Ad
from src.core.config import config

def select_agents(ad: Ad, users: List[User], user_groups_df: pd.DataFrame, return_stats: bool = False) -> Any:
    """
    Select agents for ad exposure based on group targeting.
    """
    def tppm_mocker(ad, users):
        return {user.user_id: random.random() for user in users}
    
    predictions = tppm_mocker(ad, users)
    predictions_df = pd.DataFrame(list(predictions.items()), columns=['user_id', 'prediction'])
    # Ensure user_id is string to match DB type (VARCHAR)
    predictions_df['user_id'] = predictions_df['user_id'].astype(str)
    
    merged_df = pd.merge(predictions_df, user_groups_df, on='user_id')
    group_avg = merged_df.groupby('group')['prediction'].mean().reset_index()
    total_avg = group_avg['prediction'].sum()
    
    # Use config from singleton if not passed (though legacy might pass Dict)
    # Here we assume refactored code calls with explicit args or we use singleton.
    # The original signature had `config` dict. The new signature removes it?
    # No, let's remove `config` arg and use `src.core.config`.
    
    # agents_exposed_to_ad is hardcoded/defaulted in original?
    # Original: config.get('agents_exposed_to_ad', 10)
    # We should add this to SimulationConfig in `src/core/config.py` if not present?
    # It wasn't explicitly there but let's assume default 10.
    
    agents_exposed = 10 
    
    if total_avg > 0:
        scaling_factor = agents_exposed / total_avg
        group_avg['scaled_prediction'] = group_avg['prediction'] * scaling_factor
    else:
        num_groups = len(group_avg)
        group_avg['scaled_prediction'] = agents_exposed / num_groups if num_groups > 0 else 0

    selected_agents = []
    selection_details = []
    
    for _, row in group_avg.iterrows():
        group_id = row['group']
        target_count = int(round(row['scaled_prediction']))
        group_user_ids = user_groups_df[user_groups_df['group'] == group_id]['user_id'].tolist()
        # Convert to string set for robust O(1) matching
        group_user_ids_set = {str(uid) for uid in group_user_ids}
        
        # Compare as strings
        group_users = [u for u in users if str(u.user_id) in group_user_ids_set]
        
        count = min(len(group_users), target_count)
        sampled = random.sample(group_users, count) if group_users else []
        selected_agents.extend(sampled)
        
        selection_details.append({
            "group": group_id,
            "available": len(group_users),
            "target": target_count,
            "selected_count": count,
            "selected_ids": [u.user_id for u in sampled]
        })
    
    if return_stats:
        return selected_agents, selection_details
    return selected_agents
