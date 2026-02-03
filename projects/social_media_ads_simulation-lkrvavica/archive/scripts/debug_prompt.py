import sys
import os
import json

# Add parent dir to path to import world modules
sys.path.append(os.getcwd())

from world.definitions import User, Ad
from world.engine import format_prompt, DEFAULT_PROMPT_TEMPLATE

def test_prompt_generation():
    # 1. Create Dummy User
    user = User(
        user_id="test_user_001",
        gender="F",
        age=28,
        profession="Software Engineer",
        hobby="Hiking",
        family="Single"
    )
    # Manually inject persona narrative (usually loaded from file)
    user.persona_narrative = (
        "I am a 28-year-old Software Engineer living in Austin. "
        "I love coding but sometimes find it isolating. "
        "My weekends are spent hiking the Greenbelt to disconnect. "
        "I value efficiency and sleek design."
    )
    
    # 2. Inject Daily Events
    user.daily_event_summaries = [
        "It's raining heavily today, so no hiking.",
        "A major deadline is looming for your project tomorrow."
    ]
    
    # 3. Create Dummy Ad
    ad = Ad(
        ad_id="ad_999",
        group="tech_enthusiasts",
        emotion_label="excitement",
        message_type="informational",
        visual_style="minimalist",
        num_people=0,
        people_present=False,
        people_area_ratio=0.0,
        product_present=True,
        product_area_ratio=0.5,
        object_count=1,
        object_list=["laptop"],
        dominant_element="product",
        text_present=True,
        text_area_ratio=0.2,
        avg_font_size_proxy=12,
        dominant_colors=["silver", "white"],
        brightness_category="high",
        saturation_category="low",
        hue_category="neutral",
        visual_impact=7,
        description="A sleek, silver laptop floating against a white background with the text 'Code Anywhere'."
    )
    
    # 4. Generate Prompt
    prompt = format_prompt(user, ad, current_day=42)
    
    print("="*60)
    print("GENERATED PROMPT:")
    print("="*60)
    print(prompt)
    print("="*60)
    
    # Check for key phrases
    assert "28 year old Software Engineer" in prompt
    assert "I am a 28-year-old Software Engineer" in prompt
    assert "Day 42" in prompt
    assert "raining heavily" in prompt
    assert "Code Anywhere" in prompt
    assert "[Instructions]" in prompt
    
    print("\n✅ Verification Successful: Prompt contains all expected elements.")

if __name__ == "__main__":
    test_prompt_generation()
