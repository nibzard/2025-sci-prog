
from src.services.llm import llm_client

def refine_persona_text(raw_text: str, profession: str) -> str:
    """
    Refines raw scraped text into a clean, first-person persona narrative.
    Removes web artifacts (cookies, menus) and fills in personality gaps.
    """
    prompt = f"""
    You are an expert creative writer for a simulation engine.
    
    [Input Data]
    Profession: {profession}
    Raw Profile Text: 
    "{raw_text[:4000]}"  # Truncate to safe limit

    [Task]
    Transform the raw text above into a cohesive, first-person persona narrative (300-500 words).
    
    [Requirements]
    1. **Format**: First-person ("I am...").
    2. **Cleanup**: REMOVE all website artifacts (cookies, navigation, 'contact me', emails, phone numbers).
    3. **Focus**: Focus on personality, worldview, daily struggles, and professional drive.
    4. **Privacy**: Generalize specific company names unless they add flavor. Remove real names if present, replace with generic or fictionalized self-reference.
    5. **Filling Gaps**: If the text is sparse, hallucinate plausible personality traits consistent with the profession to create a rounded character.
    6. **Tone**: Natural, engaging, human.

    [Output]
    Return ONLY the narrative text. Do not include introductory remarks.
    """
    
    # Using 'default' model from config via service
    # User requested 'google' explicitly due to OpenAI quota limits
    response = llm_client.query(prompt, backend="google", model_name="gemma-3-27b-it", json_mode=False) 
    return response.strip()

import re

def clean_persona_text_heuristic(text: str) -> str:
    """
    Fallback cleaner using regex to remove common web artifacts.
    Used when LLM service is unavailable (e.g. quota limits).
    """
    # Remove cookie warnings
    text = re.sub(r'(?i)To ensure the best experience on our website.*', '', text)
    text = re.sub(r'(?i)This website uses cookies.*', '', text)
    text = re.sub(r'(?i)we recommend that you allow cookies.*', '', text)
    
    # Remove common navigation/footer terms if at start/end
    # (Simple truncation if found in last 20% of text)
    
    # Remove URLs
    text = re.sub(r'http[s]?://\S+', '', text)
    
    return text.strip()
