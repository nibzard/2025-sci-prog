# Google Custom Search Engine Setup Guide

## Step-by-Step Instructions

### 1. Create Search Engine
- Go to: https://programmablesearchengine.google.com/controlpanel/create
- You'll need to sign in with your Google account

### 2. Configure Search Engine
**What to search:**
- Select: ☑ "Search the entire web"

**Name of the search engine:**
- Enter any name, e.g., "About.me Persona Search"

### 3. Create & Get ID
- Click the blue "Create" button
- You'll be redirected to the control panel
- Look for "Search engine ID" 
- It looks like: `0a1b2c3d4e5f6g7h8:i9j0k1l2m3`
- Click the copy icon or select and copy it

### 4. Add to .env
Open your `.env` file and add this line:
```
GOOGLE_CSE_ID=paste_your_copied_id_here
```

Make sure there are no spaces around the `=` sign.

### 5. Save & Test
- Save the `.env` file
- Run: `python harvest_only.py`
- You should see it finding profiles for most professions!

## Example .env File
```
GOOGLE_API_KEY=AIzaSy...your_existing_key
GOOGLE_CSE_ID=0a1b2c3d4e5f6g7h8:i9j0k1l2m3
OPENAI_API_KEY=sk-...your_openai_key_if_you_have_one
```

## Troubleshooting

**Error: "Invalid API key"**
- Make sure you copied the full Search Engine ID
- Check for extra spaces or quotes

**Error: "Daily limit exceeded"**
- Free tier: 100 queries/day
- You're using: ~5 queries per profession × 44 = ~220 queries
- You'll need to run it in batches or upgrade (but unlikely on first run)

**Still only finding 1 profile?**
- Make sure you saved the `.env` file
- Check the ID is on its own line
- Restart your terminal/IDE to reload environment variables
