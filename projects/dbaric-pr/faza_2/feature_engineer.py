import pandas as pd
import re

# Croatian-specific characters
CROATIAN_CHARS = set('čćšžđČĆŠŽĐ')

def count_words(text):
    """Count words in text"""
    if pd.isna(text) or not text:
        return 0
    # Split by whitespace and filter out empty strings
    words = [w for w in re.split(r'\s+', str(text)) if w.strip()]
    return len(words)

def is_croatian_word(word, context_croatian=False):
    """
    Determine if a word is Croatian based on Croatian-specific characters.
    Croatian has: č, ć, š, ž, đ (and uppercase versions)
    If context is Croatian, ambiguous words are also considered Croatian.
    """
    if not word:
        return False
    
    # Remove punctuation and check if word contains Croatian characters
    cleaned = re.sub(r'[^\w]', '', word)
    if not cleaned:
        return False
    
    # Check if word contains Croatian-specific characters
    if CROATIAN_CHARS.intersection(set(cleaned)):
        return True
    
    # Common Croatian words without special characters (common patterns)
    croatian_patterns = [
        r'^(je|su|bi|će|sam|si|smo|ste|su|sam|si|smo|ste)$',  # Common verbs
        r'^(i|ili|ali|pa|te|da|ne|na|u|za|od|do|iz|sa|s|k|o)$',  # Common prepositions/conjunctions
        r'^(koji|koja|koje|koju|kojem|kojoj|koji|koja|koje)$',  # Relative pronouns
        r'^(ovo|ova|ovo|ovaj|ova|ovo|ovog|ove|ovog|ovom|ovoj|ovim|ovima)$',  # Demonstratives
    ]
    
    cleaned_lower = cleaned.lower()
    for pattern in croatian_patterns:
        if re.match(pattern, cleaned_lower):
            return True
    
    # If context suggests Croatian and word looks like Croatian (not obviously English)
    if context_croatian:
        # Exclude obvious English words (common tech terms, etc.)
        english_tech_terms = {
            'fe', 'be', 'fs', 'qa', 'api', 'ui', 'ux', 'http', 'https', 'url', 'id', 'ids',
            'user', 'users', 'page', 'pages', 'form', 'forms', 'data', 'code', 'test', 'tests',
            'fix', 'fixes', 'bug', 'bugs', 'feature', 'features', 'update', 'updates',
            'add', 'added', 'remove', 'removed', 'change', 'changes', 'edit', 'edits'
        }
        if cleaned_lower not in english_tech_terms:
            # If it's a longer word (4+ chars) and not obviously English, likely Croatian in Croatian context
            if len(cleaned) >= 4 and cleaned.isalpha():
                return True
    
    return False

def is_english_word(word, context_croatian=False):
    """
    Determine if a word is likely English.
    A word is English if it doesn't contain Croatian characters and is clearly English.
    """
    if not word:
        return False
    
    cleaned = re.sub(r'[^\w]', '', word)
    if not cleaned:
        return False
    
    # If it has Croatian characters, it's not English
    if CROATIAN_CHARS.intersection(set(cleaned)):
        return False
    
    # Common English tech terms
    english_tech_terms = {
        'fe', 'be', 'fs', 'qa', 'api', 'ui', 'ux', 'http', 'https', 'url', 'id', 'ids',
        'user', 'users', 'page', 'pages', 'form', 'forms', 'data', 'code', 'test', 'tests',
        'fix', 'fixes', 'bug', 'bugs', 'feature', 'features', 'update', 'updates',
        'add', 'added', 'remove', 'removed', 'change', 'changes', 'edit', 'edits',
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had',
        'this', 'that', 'these', 'those', 'and', 'or', 'but', 'with', 'from', 'to', 'for'
    }
    
    cleaned_lower = cleaned.lower()
    if cleaned_lower in english_tech_terms:
        return True
    
    # If context is not Croatian and word is alphabetic ASCII, likely English
    if not context_croatian and cleaned.isalnum() and all(ord(c) < 128 for c in cleaned):
        return True
    
    return False

def analyze_language(text):
    """
    Analyze text and return word counts and percentages for Croatian and English.
    Returns: (total_words, croatian_words, english_words, croatian_pct, english_pct)
    """
    if pd.isna(text) or not text:
        return 0, 0, 0, 0.0, 0.0
    
    # Split text into words
    words = re.findall(r'\b\w+\b', str(text))
    
    if not words:
        return 0, 0, 0, 0.0, 0.0
    
    # First pass: identify words with Croatian characters to determine context
    words_with_croatian_chars = sum(1 for w in words if CROATIAN_CHARS.intersection(set(re.sub(r'[^\w]', '', w))))
    context_croatian = words_with_croatian_chars > 0
    
    # Second pass: classify words with context
    total_words = len(words)
    croatian_words = sum(1 for w in words if is_croatian_word(w, context_croatian))
    english_words = sum(1 for w in words if is_english_word(w, context_croatian))
    
    # Calculate percentages
    croatian_pct = (croatian_words / total_words * 100) if total_words > 0 else 0.0
    english_pct = (english_words / total_words * 100) if total_words > 0 else 0.0
    
    return total_words, croatian_words, english_words, croatian_pct, english_pct

# Read the CSV
df = pd.read_csv('source_v2.csv')

print("Analyzing descriptions...")
# Analyze description
description_results = df['description'].apply(analyze_language)
df['description_word_count'] = [r[0] for r in description_results]
df['description_croatian_words'] = [r[1] for r in description_results]
df['description_english_words'] = [r[2] for r in description_results]
df['description_croatian_pct'] = [r[3] for r in description_results]
df['description_english_pct'] = [r[4] for r in description_results]

print("Analyzing titles...")
# Analyze title
title_results = df['title'].apply(analyze_language)
df['title_word_count'] = [r[0] for r in title_results]
df['title_croatian_words'] = [r[1] for r in title_results]
df['title_english_words'] = [r[2] for r in title_results]
df['title_croatian_pct'] = [r[3] for r in title_results]
df['title_english_pct'] = [r[4] for r in title_results]

# Save to source_v2.csv (overwrite)
df.to_csv('source_v2.csv', index=False)

print(f"✅ Added language analysis features to {len(df)} rows")
print(f"   - Description: word count, Croatian/English word counts and percentages")
print(f"   - Title: word count, Croatian/English word counts and percentages")
print(f"   Saved to source_v2.csv")

