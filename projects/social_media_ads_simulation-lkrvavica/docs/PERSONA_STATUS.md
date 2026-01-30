# Persona System - Current Status & Next Steps

## What's Working ✅
- Harvesting infrastructure is complete
- Scoring simplified to "longest bio wins"
- Playwright scraping captures full content
- User class loads personas automatically
- 1/44 professions successfully harvested ("writer")

## The Blocker 🚧

**The harvester needs search results to scrape.**

Currently using `ManualSearchProvider` which only has URLs for 2 jobs:
- writer ✅
- carpenter (but URL might be invalid)

For the other 42 professions, it has no URLs → nothing to scrape.

## Solution Options

### Option 1: Google Custom Search (Recommended - 5 min setup)

**Setup:**
1. Visit: https://programmablesearchengine.google.com/controlpanel/create
2. Select "Search the entire web"
3. Name it anything (e.g., "Persona Search")
4. Click Create
5. Copy the "Search engine ID" 
6. Add to `.env`: `GOOGLE_CSE_ID=your_id_here`
7. Run: `python harvest_only.py`

**Result:** Automatically searches Google for all 44 professions

**Free tier:** 100 queries/day (more than enough)

### Option 2: Manual URL Collection (Tedious)

Edit `data/persona_raw/manual_discovery.json` and add URLs for all 44 jobs:

```json
{
  "writer": ["https://about.me/jeffgoins"],
  "carpenter": ["https://about.me/james_pen"],
  "artist": ["https://about.me/artist1", "https://about.me/artist2"],
  "baker": ["https://about.me/baker1"],
  ... (40 more)
}
```

Then run: `python harvest_only.py`

### Option 3: Use What We Have (Testing Only)

- Keep just the "writer" persona
- Test the full system end-to-end
- Expand later

## Once Harvesting Works

After you get 30-40+ successful harvests:

1. **Check the data:**
   ```bash
   python -c "import json; lines=open('data/persona_raw/job_best_profile.jsonl').readlines(); print(f'{len(lines)} personas harvested')"
   ```

2. **Generate narratives:**
   - Fix Google API issue (or use OpenAI)
   - Run: `python generate_all_personas.py`
   - Skip to synthesis phase

3. **Integrate into simulation:**
   - Already done! User class loads personas automatically
   - Test with visualizer

## Recommendation

**Set up Google CSE** - it's the fastest path to a working system. Takes 5 minutes and automates everything.

Let me know which option you'd like to pursue!
