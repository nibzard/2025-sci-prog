# PERSONAS - Collect 1 High-Quality about.me Profile per Job (Search API → Steel)

## 0) Goal

For each first in user's profession list, collect exactly one public about.me profile that contains a usable first-person bio. Store the selected profile and enough metadata to reproduce/diagnose the selection later.

The output will be used as `persona_raw_source` for later persona synthesis, so the selected bio should be non-trivial (ideally >150 words) and contain some personal context (interests, motivations, routine, etc.). It will not be permanently stored, it will just be used in another phase. 

## 2) System Overview

The system has two phases per job:

1. **Search Phase (Search API)**  
   Collect a small set of candidate URLs from search results for the given job query.

2. **Scrape + Select Phase (Steel)**  
   For each candidate URL, scrape minimal page signals and compute a quality score.  
   Select the highest-scoring profile as the job's representative.

## 3) Search Phase (API)

### 3.1 Search provider

Use a Search API that returns results as json: (preferred: Bing Web Search API or equivalent). The system must not scrape Google SERPs directly.

### 3.2 Query templates

For each job, generate queries in priority order:

- **Q1 (high-quality bios):**  
  `site:about.me "<JOB>" ("I love" OR "my passion" OR "my story")`

- **Q2 (fallback – first-person baseline):**  
  `site:about.me "<JOB>" ("I am" OR "I'm")`

- **Q3 (fallback – broad):**  
  `site:about.me "<JOB>"`

**Rules:**

- Attempt Q1 first.
- If after filtering there are < `target_candidates_per_query` valid candidates, attempt Q2.
- If still insufficient, attempt Q3.

### 3.3 Result count + pagination

For each query, request up to `target_candidates_per_query` results.  
If the API returns fewer results, accept and proceed to next query type.

### 3.4 URL filtering (must)

From returned results, keep only URLs that look like about.me profiles.

**Accept if:**

- host is `about.me` (case-insensitive)
- path has exactly one segment: `/username`
  - `segments = path.strip("/").split("/")`
  - `len(segments) == 1`
- the segment is not in a blacklist (see below)

**Blacklist segments:**
["about", "privacy", "terms", "help", "support", "blog", "press", "contact", "login", "signup", "pricing", "search"]
(Extend as needed.)

**Canonicalize URLs:**

- strip query params and fragments
- normalize scheme to `https`
- remove trailing slash

### 3.5 Deduplication

Deduplicate by canonical URL across all queries/aliases for that job.

### 3.6 Output of Search Phase

For each job, write candidate URLs into storage:

**`search_candidates.jsonl`** (one row per candidate):
run_id
job_primary
job_term_used (job or alias that generated this candidate)
query_used (Q1/Q2/Q3 string)
url
rank (rank in returned results if available)
retrieved_at (timestamp)

## 4) Scrape + Select Phase (Steel)

### 4.1 Steel usage

Use Steel to load each candidate URL in a real browser session and extract content reliably (including JS-rendered pages if applicable). Steel is used for profile scraping, not for search.

### 4.2 Extracted fields per profile

For each candidate URL, extract the following signals (in priority order):

- `og_description` = content of `meta[property="og:description"]`
- `meta_description` = content of `meta[name="description"]`
- `page_title` = `<title>` text (optional)
- `h1_text` = first visible `<h1>` text (optional)
- `visible_text_fallback` = visible text from main content (capped, e.g., first 1,000–2,000 characters)

**Chosen bio text should be selected with:**

- if `og_description` exists and is non-empty → use it
- else if `meta_description` exists → use it
- else use `visible_text_fallback`

### 4.3 Data sanitization (mandatory)

Before scoring/storing bio text:

- remove emails (regex)
- remove phone numbers (regex)
- remove obvious @handles (regex)
- remove excessive whitespace
- do not store personal names as "persona names" (but you can keep h1/title as raw fields if needed for debugging; do not use later in final persona)

### 4.4 Quality scoring (must)

Compute `quality_score` for each candidate. The goal is to select a bio that is:

- long enough to be meaningful
- written in first person
- contains some personal detail beyond job title
- not spammy/link-only

**Scoring rubric (implement exactly):**

Let:

- `bio` = chosen_bio_text
- `word_count` = len(bio.split())
- Initialize `score` = 0

**Length:**

- if `word_count >= 150`: score += 3
- else if `word_count >= 100`: score += 2
- else if `word_count >= 50`: score += 1
- else: score -= 3

**First-person signal:**

- if bio contains any of `[" I ", " I'm", " I'm", " my ", " me "]` (case-insensitive): score += 2
- else: score -= 1

**Life-detail signals (at least one):**  
Add +2 if bio contains any keyword from:

- routine/time: `["shift", "weekend", "commute", "evening", "morning", "after work", "late night"]`
- family/life: `["parent", "mom", "dad", "kids", "family", "partner"]`
- learning/status: `["student", "studying", "degree", "course", "training", "certified"]`
- interests: `["hobby", "I love", "passion", "music", "sports", "travel", "reading", "gaming", "cooking", "fitness", "photography"]`

(Only one +2 even if multiple keywords.)

**Spam/generic penalties:**

- if bio contains "contact me" / "get in touch" / "hire me" and `word_count < 30`: score -= 2
- if bio has > 3 URLs or looks like link list: score -= 3
- if bio is mostly hashtags or keyword soup: score -= 3
- if bio is empty or boilerplate ("Welcome to my page"): score -= 3

### 4.5 Selection rule

Choose the candidate with the highest `quality_score`.

**Tie-breakers:**

1. higher `word_count`
2. earlier rank in search results (lower rank number)
3. stable deterministic tie-break (e.g., lexicographically smallest URL) for reproducibility

### 4.6 Minimum quality threshold + fallback

Define `MIN_ACCEPTABLE_SCORE = 2`.

If the best candidate has `quality_score < MIN_ACCEPTABLE_SCORE`:

- re-run Search Phase for that job using:
  - next query type (if not tried)
  - next alias (if available)
  - increase candidates per query (e.g., 20 → 50) up to a cap
- then repeat scraping and selection

If still below threshold after exhausting query budget:

- accept the best available but mark it as `low_quality=true`

### 4.7 Output of Scrape + Select Phase

Write per-candidate scraped results (optional but recommended for debugging):

**`profile_candidates_scored.jsonl`**:
run_id, job_primary, url
bio_text, word_count
quality_score
signals_present (booleans: first_person, life_detail, spam_flag)
scraped_at
source_query_used

Write final selected per-job profile:

**`job_best_profile.jsonl`** (one row per job):
run_id
job_primary
selected_url
selected_bio_text
word_count
quality_score
selection_reason (brief string, e.g. "highest score; first-person; >40 words")
query_used (which query produced it)
job_term_used (job or alias)
candidates_considered_count
low_quality (bool)
timestamp

## 5) Non-functional Requirements

### 5.1 Polite scraping

- rate-limit Steel page loads (configurable)
- implement retry with exponential backoff for transient failures (timeouts, 5xx)
- if blocked, record `blocked=true` and retry later

### 5.2 Reproducibility

- all decisions should be deterministic given:
  - input jobs list
  - search results returned
  - scoring rubric
- tie-breakers must be deterministic

### 5.3 Logging

For each job, log:

- how many candidates found per query
- how many filtered out (non-profile URLs)
- how many successfully scraped
- why selection succeeded/failed

## 6) Definition of Done

A run is considered successful if:

- for ≥ 90% of jobs, a selected profile is produced with `quality_score >= MIN_ACCEPTABLE_SCORE`
- outputs (`search_candidates.jsonl`, `job_best_profile.jsonl`) are complete and consistent
- rerunning the pipeline (same jobs, similar API results) produces stable selections

## 7) Notes / Next Step (outside this spec)

This spec stops at "1 best profile per job". The next pipeline stage (persona synthesis into `persona_long` / `persona_compact`) will consume `job_best_profile.jsonl` as a single-source `persona_raw_source`.

# CALENDAR - Spec: Events Module (Calendar + Random Pools + Deterministic Selection)
0) Goal

Generate per-day events for a simulation with:

Global events: 0–1/day (plus fixed events like payday that may co-exist)

Group events: 0–1/day (only for some groups)

Personal events: 1–2/day (for variability)

The module must support:

A 100-day calendar year (5 months × 20 days)

Fixed scheduled events (e.g., payday) and additional scheduled global events (holidays, weather periods)

Random "unpredictable" event pools (global/group/personal)

Deterministic selection driven by a seed

Optional multi-day events via duration_days

The output of the module is the list of event cards assigned to each agent for each day.

1) Inputs
1.1 Simulation config

experiment_id: str

total_days: int = 100

months: int = 5

days_per_month: int = 20

1.2 Agent & group info

agents: List[Agent] where each agent has:

agent_id: str

group_id: str (agents already divided into groups)

1.3 Event libraries

A versioned library file (or DB table) containing:

global_pool_events: List[EventCard] (target ~20)

group_pool_events: List[EventCard] (target ~20)

personal_pool_events: List[EventCard] (target ~60–100)

1.4 Calendar definition

A 100-day calendar definition with:

fixed events (payday schedule)

scheduled holidays

scheduled weather shifts

optional other scheduled global events

Calendar is simulated, not tied to real-world dates.

2) Event Card Schema (Library Format)

Each event must be an "event card" (not a narrative story). Required fields:

json:
  "event_id": "PERS_BREAKUP_02",
  "scope": "personal",                    // global | group | personal
  "text": "You had a messy breakup yesterday and slept poorly.",
  "tags": ["relationship", "sleep", "stress"],
  "severity": 0.7,                        // 0.0–1.0
  "polarity": "negative",                 // positive | neutral | negative
  "frequency_weight": 0.6,                // sampling weight, >= 0.0
  "cooldown_days": 14,                    // minimum days before repeat for same target
  "duration_days": 1,                     // >=1; if >1, event persists across days
  "expected_effects_hint": ["patience_down", "attention_down"]


Notes:

expected_effects_hint is for analysis/debugging and can be excluded from model prompt later.

cooldown_days applies per-target:

global: applies globally

group: applies per group

personal: applies per agent

duration_days controls how long the event remains active (see Section 5).

3) Calendar Specification
3.1 Year layout

Simulation year = 100 days

5 months × 20 days

Represent each day as:

day_index: int in [1..100]

month_index: int in [1..5]

month_day: int in [1..20]

3.2 Fixed events (payday)

Payday is a fixed global event that repeats:

Every 20 days, starting from day 5 

That means: day 5, 25, 45, 65, 85

Payday can co-exist with another global event on the same day.

3.3 Other scheduled global events

Calendar also contains:

2–3 holidays across 100 days (scheduled on fixed day indices)

5–10 weather "shifts" across 100 days

each shift can have duration_days (recommended 2–7 days)

Calendar events are stored as event cards too, with scope="global" and a special tag ["calendar"].

3.4 Daily global event quota

Per day:

scheduled fixed events may occur (e.g., payday)

additionally, there may be 0–1 non-fixed global event (either scheduled or random)

Rule:

If the calendar defines a scheduled non-fixed global event for that day, it is used.

Else the module may draw a random global event from the global pool (0–1/day) depending on selection rule (Section 4).

4) Deterministic Event Selection Rules (Detailed)
4.1 Deterministic RNG

For every selection, create a deterministic pseudo-random generator from a seed string.

Base seed string:

seed_base = f"{experiment_id}|{day_index}|{scope}|{target_id}"

Where:

scope ∈ {"global","group","personal"}

target_id is:

global: "GLOBAL"

group: group_id

personal: agent_id

Convert seed_base into a stable integer seed using a stable hash (do NOT use language default hash if it changes between runs; use SHA256 → int).

Example deterministic seed:

seed_int = int(sha256(seed_base).hexdigest()[:16], 16)

Use seed_int to initialize RNG (language-specific).

4.2 Candidate filtering pipeline

When selecting an event from a pool (global/group/personal), apply filters in this order:

Audience filters (if implemented later) — ignore for now unless already needed.

Duration carryover check

If there is an active carryover event for this scope+target from a previous day (Section 5), include it automatically and do not roll a new one for that slot unless quota requires additional events.

Cooldown filter
Remove any event where:

last occurrence day for this target is within cooldown_days

definition: day_index - last_day_used[target, event_id] < cooldown_days

Result: filtered_candidates.

If filtered_candidates is empty:

relax cooldown by ignoring it (or pick the least-recently-used event) as a fallback.

mark fallback mode in output metadata (optional).

4.3 Weighted selection

From filtered_candidates, pick one event using frequency_weight.

Algorithm:

compute total weight W = sum(event.frequency_weight)

sample u = RNG.uniform(0, W)

iterate cumulative sum until u crossed

Edge cases:

if all weights are 0, treat as uniform sampling.

4.4 Daily quotas (exact behavior)

For each day_index:

Global

Add all scheduled fixed global events (e.g., payday) if today matches schedule.

Then decide whether to add 0–1 additional global event:

If calendar has a scheduled global event today (holiday/weather start): add it.

Else optionally sample one from global pool with a probability p_global (default 0.5) OR always 0 unless you need more dynamics.

If sampled, apply selection rules above with target_id "GLOBAL".

Group (per group)

For each group, decide if it gets a group event today:

quota: 0–1 group event/day for that group

use deterministic RNG with target_id = group_id

probability p_group (default 0.3) or configured

if triggered, sample one from group_pool_events

Assign resulting group events to all agents in that group.

Personal (per agent)

Each agent gets 1–2 personal events/day:

Always select 1 personal event (mandatory).

Select a second personal event with probability p_personal_extra (default 0.35) for variability.

Use deterministic RNG with target_id = agent_id and include an index in seed for the "slot":

slot1 seed: ...|personal|agent_id|slot=1

slot2 seed: ...|personal|agent_id|slot=2

Selection for each slot uses the filtering + weighted selection pipeline.

4.5 Output guarantee

The module must ensure that per agent/day:

personal events list length is 1 or 2

global events list length is >=0 (plus possible payday)

group event list length is 0 or 1 (based on group)

5) Duration (Multi-day events)
5.1 Meaning

If duration_days > 1, then once selected, the event remains active for subsequent days.

Example:

weather event "heavy snow" with duration 3 days

personal event "mild cold" duration 2 days

5.2 Carryover mechanism

Maintain an "active events" state:

active_events_global: List[ActiveEvent]

active_events_group[group_id]: List[ActiveEvent]

active_events_personal[agent_id]: List[ActiveEvent]

Where ActiveEvent includes:

event_id

start_day

end_day = start_day + duration_days - 1

On each day:

before selecting new events, include all ActiveEvents whose start_day <= day_index <= end_day

do not re-sample the same scope+target event if it is already active

cooldown should treat active periods as "used" across the span

5.3 Re-selection after duration ends

After an event expires, it becomes eligible again only after cooldown passes.

6) Event Content Creation (Library Building) — Option A
6.1 Approach

Use a hand-defined event taxonomy (templates) and an LLM to generate multiple variants.

Process:

Manually define taxonomy categories per scope:

Global: weather, public mood, platform changes, seasonal routines, holidays

Group: team conflict, group outing, group deadline, shared celebration

Personal: sleep, health, money, relationships, chores, minor wins/losses, commute

For each category define 3–10 "seed templates" (short prompts).

Ask LLM to generate variants:

each variant must fill the event card schema

vary severity/polarity, but keep realism

ensure text is short and concrete (1 sentence)

Human review pass:

remove nonsense/duplicates

ensure tag diversity

set frequency_weight and cooldown_days sensibly

Freeze into events_library_vX.json.

6.2 Example taxonomy template (personal)

Template: "sleep disruption"

Variants LLM should produce:

"You slept 4 hours and feel foggy."

"You woke up multiple times and feel restless."

"You overslept and feel rushed."
Each with:

tags: ["sleep", "energy"]

polarity: neutral/negative

severity: 0.2–0.6

duration_days: 1–2

cooldown_days: 5–10

frequency_weight: moderate (0.7–1.2)

7) Outputs
7.1 Per-day event assignment

For each agent and day, output a structure:

json:
  "experiment_id": "...",
  "day_index": 12,
  "agent_id": "A_032",
  "group_id": "G_04",
  "global_events": [EventCard...],
  "group_events": [EventCard...],
  "personal_events": [EventCard...]


EventCard objects included in output should contain at least:

event_id, scope, text, tags, severity, polarity
Optional for internal storage:

expected_effects_hint, cooldown_days, duration_days, frequency_weight

7.2 Stability requirement

Given the same:

experiment_id

agent ids + group mapping

event library version

calendar definition

…the module must generate the same assignments.

8) Logging & Debugging

(TBD)

9) Definition of Done

Produces event assignments for 100 days for all agents.

Per day quotas respected:

global: scheduled + optional 0–1 extra

group: 0–1 per group

personal: 1–2 per agent

Deterministic output for the same inputs.

Supports duration_days carryover.

Uses taxonomy + LLM generation workflow to build the library.
# RESPONSE
Spec: Output Format – Free Text + Action Extraction (Fault-Tolerant)
0) Goal

Allow the model to respond in natural language, while still enabling robust, non-brittle extraction of interaction signals (IGNORE / CLICK / LIKE / DISLIKE / SHARE / combinations).

The system must not fail if the model:

forgets the tag

slightly misspells the tag

mentions actions earlier in the text

uses natural language instead of the exact tag

Constraints are soft at generation time and hard only at analysis time.

1) Expected Output (Soft Contract)
1.1 Preferred format (but not guaranteed)

Total length: 3–5 sentences

Content structure (guideline, not enforced):

1–2 sentences: subjective experience / reaction

1 sentence: interpretation of events/context

Last sentence: explicit interaction declaration

1.2 Preferred interaction marker

The model is instructed (but not forced) to end with:

Action: IGNORE | CLICK | LIKE | DISLIKE | SHARE | CLICK+LIKE | CLICK+SHARE | DISLIKE+CLICK | etc.


Examples:

Action: IGNORE

Action: CLICK+LIKE

Action: CLICK+SHARE

This is not required, only preferred.

2) Hard Requirement (System-side)

The program must not break if the last sentence is not in this format.

Therefore:

No assumptions that the tag exists

No assumptions that it is the last sentence

No assumptions that spelling/casing is exact

3) Action Extraction Pipeline (Deterministic, Multi-step)

Action extraction happens after generation, in analysis code.

Step 1: Attempt strict tag parse (happy path)

Regex (case-insensitive):

Action:\s*([A-Z+]+)


Valid tokens:

IGNORE

CLICK

LIKE

DISLIKE

SHARE

Split on + to get multiple actions.

If successful:

mark extraction_mode = "explicit_tag"

done

Step 2: Fuzzy tag recovery (common model errors)

If Step 1 fails, try relaxed patterns:

Action - CLICK

Action CLICK

Actions: CLICK and LIKE

Final action: CLICK+LIKE

Regex examples:

Action[s]?:?\s*([A-Z+\sand]+)
Final action:?\s*(...)


Normalize:

map "and" → +

uppercase

filter to valid action tokens

If ≥1 valid action found:

extraction_mode = "fuzzy_tag"

done

Step 3: Natural language inference (fallback)

If no explicit action tag is found:

Run a lightweight inference pass over the full text:

Rules-based (fast, deterministic):

If text contains:

"I ignore", "I scroll past", "I move on" → IGNORE

"I click", "I tap", "I open" → CLICK

"I like", "I heart" → LIKE

"I dislike", "I'm annoyed and mark it" → DISLIKE

"I share", "I send it to a friend" → SHARE

Multiple matches → MULTI.

If still ambiguous:

mark action as UNKNOWN

extraction_mode = "unresolved"

This is acceptable and expected for outlier analysis.

4) Output of the Extraction Step (Internal Representation)

Regardless of how extraction succeeds, normalize into:

json:
  "actions": ["CLICK", "LIKE"],
  "extraction_mode": "explicit_tag | fuzzy_tag | inferred | unresolved",
  "confidence": 0.0–1.0


Confidence heuristics:

explicit_tag → 0.95

fuzzy_tag → 0.75

inferred → 0.4

unresolved → 0.0

This confidence is critical for later analysis.

5) Compatibility Rules (Moved to Analysis Phase)

Your original compatibility constraints (e.g. IGNORE incompatible with SHARE) are not enforced at generation time.

Instead:

After extraction, run a compatibility validator:

flags incompatible combos

does not discard the sample

labels it as outlier_reason = "action_incompatibility"

This aligns perfectly with your stated goal: find and inspect outliers, not prevent them.

6) Failure Tolerance & Guarantees

The system must guarantee:

❌ No runtime exception if:

tag is missing

tag is malformed

last sentence is not an action

✅ Every response produces:

a stored raw text

a parsed action object (even if UNKNOWN)

an extraction_mode

This ensures:

zero data loss

full auditability

traceability of "model drift" and "confabulation"

7) Prompt Instruction (Minimal, Non-Technical)

In the prompt, include only this, nothing stricter:

"Write 3–5 sentences describing your reaction.
If possible, end the last sentence with an explicit action such as:
Action: CLICK, Action: IGNORE, or combinations like Action: CLICK+LIKE."

Do not mention parsing, regex, or constraints.

8) Why this design fits your research goals

This setup:

allows stylistic freedom (no json: jail)

preserves interpretability

intentionally allows errors so you can:

detect drift

detect confabulation

detect misalignment between text and action

gives you quantitative signals (extraction_mode, confidence) for filtering and analysis

Most importantly:

Parsing robustness is your safety net, not the model's responsibility.

9) Definition of Done

≥ 95% of responses yield a parsed action (explicit or inferred)

0% pipeline crashes due to malformed output

All responses stored verbatim

Action extraction is deterministic and reproducible
# PROMPT Spec: Prompt Design – Single-Prompt Interaction 
0) Goal

For each simulated interaction, generate a natural-language reaction to an ad that:

reflects the agent's persona and current events,

remains interpretable and auditable,

produces an extractable interaction signal (IGNORE / CLICK / LIKE / DISLIKE / SHARE / combinations),

does not rely on persistent conversational state.

1) Inputs to the Prompt (Per Request)

Each request must include all relevant state explicitly, in the following order.

1.1 Persona (compact)

Provide the agent's current persona in compact form:

8–12 bullet points describing attitudes, habits, preferences

5–10 numeric personality metrics (e.g. patience, skepticism, impulsivity)

Optional:

a small persona patch (1–2 bullets) if this agent was adjusted due to prior analysis

Persona text must:

be self-contained

not reference previous days or past interactions

not assume memory across prompts

1.2 Events (current day)

Provide only the events active on the current day.

Events are supplied as plain text, derived from event cards, grouped by scope:

Global events (0–1, plus fixed if applicable)

Group events (0–1, only if the agent's group has one)

Personal events (1–2)

Each event should be rendered as:

a single sentence

no IDs or metadata

no explicit severity numbers

Example format:

Global events:
- It has been raining heavily for two days, making everything feel slower.

Group events:
- Your team canceled a planned outing due to low turnout.

Personal events:
- You slept poorly and feel slightly irritable.
- You received an unexpected small expense today.

1.3 Ad description

Provide the ad as a natural-language description, not raw json:.

Guidelines:

2–5 sentences

describe visuals, tone, product/category, and implied message

no platform-specific technical metadata

1.4 Output instructions

Provide minimal, non-technical guidance on how to respond.

Required instructions:

response length

style

action hint (soft)

2) Prompt Assembly Rules
2.1 Ordering

The prompt must follow this order:

Persona

Events

Ad description

Output instructions

This ordering is mandatory to ensure consistent interpretation.

2.2 Required soft instructions (verbatim or near-verbatim)

The following ideas must be present in the prompt:

Freedom of interpretation:

"You may interpret the listed events freely. If you invent additional circumstances, make them plausible and brief."

Anti-pattern pressure:

"Try not to repeat the same reasoning style across different days."

These are soft constraints and must not be framed as rules or checks.

3) Output Expectations (Soft Contract)

The model is requested, but not forced, to produce:

3–5 sentences total

1–2 sentences describing subjective experience or reaction

1 sentence reflecting how events influenced mood/attention

Final sentence explicitly stating the interaction

3.1 Preferred interaction marker

The final sentence should, if possible, end with:

Action: IGNORE | CLICK | LIKE | DISLIKE | SHARE | combinations (e.g. CLICK+LIKE)


This is not a hard requirement.

4) Hard System Requirement

The system must not fail if the output does not follow the preferred format.

Specifically:

the last sentence may not contain an action

the action may appear earlier in the text

the model may use natural language instead of a tag

All such cases must be handled by the downstream extraction logic (see Action Extraction spec).

5) Things Explicitly NOT Included in the Prompt

The prompt must NOT:

mention parsing, regex, or extraction

mention compatibility rules between actions

mention "outliers", "analysis", or "validation"

enforce json: or rigid schemas

reference prior days or memory

6) Determinism & Reproducibility

For a given:

persona

events

ad description

prompt template version

…the generated response should be treated as a single atomic sample.

The system must:

store the full prompt text

store the full raw response text

associate both with a unique interaction ID

7) Notes: Alternative Prompting Strategy
7.1 Alternative (Not Implemented Here): Two-Round Prompting

An alternative design exists in which:

the first prompt synthesizes a "current internal state" from persona + events

the second prompt uses that state to react to the ad

This approach can increase internal coherence but:

introduces an additional confabulation layer

makes outlier attribution harder

increases system complexity and cost

Option 1 (single-prompt) is the default and required implementation.
The two-round approach may be explored later as an experimental or comparative mode, but is out of scope for this spec.

8) Definition of Done

This module is complete when:

prompts are generated deterministically from inputs

responses are natural language (not structured json:)

no downstream component depends on perfect formatting

all prompts and responses are fully logged and replayable

action extraction can be applied robustly to all outputs