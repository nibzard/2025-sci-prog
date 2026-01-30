# Persona System - Quick Reference

## Overview
The persona system enhances agent realism by replacing basic demographics with rich 300-500 word narrative profiles sourced from real about.me bios.

## Files Created/Modified

### New Files
- `world/persona_harvester.py` - Searches and scores about.me profiles
- `world/persona_generator.py` - Synthesizes 300-500 word narratives via LLM
- `generate_all_personas.py` - Batch generation script for all professions
- `data/persona_raw/job_best_profile.jsonl` - Harvested raw bios
- `data/persona_synthesized/*.txt` - Generated persona narratives
- `data/persona_mapping.json` - Job → file path lookup

### Modified Files
- `world/definitions.py` - User class now loads personas automatically
- `world/engine.py` - Added `json_mode` parameter to `query_llm()`

## How to Use

### Generate All Personas (After Fixing API Keys)
```bash
python generate_all_personas.py
```

### Test Persona Loading
```python
from world.definitions import User
u = User(1, 'm', 25, "['writer']", "['reading']", 'y')
print(u.persona_narrative)  # Should show 300-500 word narrative
```

### Disable Personas (If Needed)
Comment out in `world/definitions.py`:
```python
# self.load_persona()  # Disable persona loading
```

## Current Status

✅ **Working:**
- Playwright-based scraping
- Persona loading in User class
- Data persistence
- Backward compatibility

⚠️ **Needs Attention:**
- API keys for automated synthesis
- Full bio content (currently ~50 words)

## API Key Setup

### For Persona Synthesis (Required)

You **already have this** in your `.env` file! The same key you use for the simulation:

```
GOOGLE_API_KEY=your_existing_key  # Same key for Gemini LLM
```

This is all you need for `persona_generator.py` to work.

### For Automated Web Search (Optional)

Only needed if you want automated about.me searching (most users won't need this):

```
GOOGLE_CSE_ID=your_custom_search_id  # For Google Custom Search API (different service)
```

**Note:** The harvester works fine without this using `ManualSearchProvider`

## Troubleshooting

**Problem:** Personas not loading
**Solution:** Check that `data/persona_mapping.json` exists and paths are correct

**Problem:** API synthesis fails
**Solution:** Verify API keys in `.env` and check quota

**Problem:** Short bios (<100 words)
**Solution:** LLM will expand them during synthesis phase

## Next Steps

1. Fix API keys
2. Run `python generate_all_personas.py`
3. Verify all 44 personas generated
4. Test simulation with personas enabled
5. Compare response quality vs. non-persona baseline
