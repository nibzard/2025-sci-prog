# Tinker Implementation Guide for Social Media Ads Simulation

## Overview

This guide provides a comprehensive implementation plan for integrating Tinker into the Social Media Ads Simulation project. By fine-tuning specialized agent models, you can replace expensive GPT-4-mini API calls with cost-effective, domain-specialized models while improving behavioral realism through reinforcement learning.

---

## Why Tinker for This Project?

### Current Challenge
The baseline design calls for **GPT-4-mini API calls for every agent decision**:
- 100 agents × 100 days × 10 ad exposures/day = **100,000 LLM calls**
- At ~$0.50-1.00 per 1M tokens (GPT-4-mini pricing)
- With ~500 tokens per call: **50M tokens = $25-50** (conservative estimate)
- **Latency**: 1-3 seconds per API call = hours of simulation time
- **Rate limits**: Potential throttling with concurrent requests

### Tinker Solution Benefits

✅ **10-100x Cost Reduction**: Fine-tuned model costs ~$0.0001 per inference
✅ **10-100x Faster**: Local inference in 0.1-0.5 seconds
✅ **Domain Specialization**: Agents learn realistic ad response patterns
✅ **RLVR Optimization**: Reward realistic, persona-consistent behavior
✅ **Scalability**: Run 1000+ agents without cost explosion
✅ **Persona-Specific Models**: Different LoRA adapters for user clusters

---

## Architecture Overview

```
┌────────────────────────────────────────────────────────────────┐
│         Social Media Ads Simulation + Tinker                   │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐│
│  │   Phase 1:   │ ───> │   Phase 2:   │ ───> │   Phase 3:   ││
│  │  Baseline    │      │ Supervised   │      │  RLVR with   ││
│  │  Simulation  │      │  Learning    │      │  Realism     ││
│  └──────────────┘      └──────────────┘      └──────────────┘│
│         │                      │                      │        │
│         │                      │                      │        │
│  Collect synthetic      Fine-tune on         Reward realistic │
│  interaction data       agent behaviors      engagement        │
│  (GPT-4-mini)          (Qwen/Llama)         patterns          │
│                                                                │
│  ┌────────────────────────────────────────────────────────┐  │
│  │          Persona-Specific LoRA Adapters                │  │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ │  │
│  │  │ Sports   │ │  Tech    │ │ Passive  │ │Influencer│ │  │
│  │  │Enthusiast│ │Enthusiast│ │  Users   │ │  Persona │ │  │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘ │  │
│  └────────────────────────────────────────────────────────┘  │
│         │                                                      │
│         ▼                                                      │
│  ┌────────────────────────────────────┐                      │
│  │   Agent-Based Simulation Loop      │                      │
│  │   (DuckDB logging, ML prediction)  │                      │
│  └────────────────────────────────────┘                      │
└────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Baseline Simulation (Data Collection)

### 1.1 Install Dependencies

```bash
# Core dependencies
pip install duckdb pandas numpy scikit-learn matplotlib seaborn

# LLM API (for baseline)
pip install openai  # or anthropic for Claude

# Chess for testing (optional)
pip install pyyaml

# Later for Tinker
pip install git+https://github.com/thinking-machines/tinker-sdk
pip install git+https://github.com/thinking-machines/tinker-cookbook
```

### 1.2 Implement Baseline Agent (Using GPT-4-mini)

```python
# agents/agent.py

import openai
from typing import Dict, List
import json

class AdAgent:
    """
    Agent representing a social media user with personality traits
    """
    def __init__(self, agent_id: str, features: Dict):
        self.agent_id = agent_id

        # Personal features
        self.age = features.get("age")
        self.gender = features.get("gender")
        self.family = features.get("family")
        self.hobbies = features.get("hobbies", [])
        self.profession = features.get("profession")

        # Propensity features (0-100%)
        self.activity_level = features.get("activity_level", 50)
        self.risk_tolerance = features.get("risk_tolerance", 50)
        self.social_engagement = features.get("social_engagement", 50)

        # Interaction history
        self.history = []

    def get_profile_summary(self) -> str:
        """Generate natural language profile for LLM prompt"""
        return f"""User Profile:
- Age: {self.age}
- Gender: {self.gender}
- Family: {self.family}
- Hobbies: {', '.join(self.hobbies)}
- Profession: {self.profession}

Personality Traits:
- Activity Level: {self.activity_level}% (how often you engage with content)
- Risk Tolerance: {self.risk_tolerance}% (willingness to try new things)
- Social Engagement: {self.social_engagement}% (tendency to share content)
"""

    def get_history_summary(self, limit=10) -> str:
        """Get recent interaction history"""
        if not self.history:
            return "No previous ad interactions."

        recent = self.history[-limit:]
        summary = []
        for interaction in recent:
            ad = interaction['ad']
            actions = interaction['actions']
            summary.append(
                f"- Ad about {ad['theme']} ({ad['rhetorical_triangle']} appeal): "
                f"{'Clicked' if actions['click'] else ''} "
                f"{'Liked' if actions['like'] else ''} "
                f"{'Shared' if actions['share'] else ''} "
                f"{'Disliked' if actions['dislike'] else ''} "
                f"{'Ignored' if actions['ignore'] else ''}"
            )

        return "\n".join(summary)

    def decide_action_baseline(self, ad: Dict, temperature=0.7) -> Dict[str, bool]:
        """
        Use GPT-4-mini to decide how to react to an ad

        Returns: Dict of actions (click, like, dislike, ignore, share)
        """
        prompt = f"""{self.get_profile_summary()}

Recent Ad Interactions:
{self.get_history_summary()}

---

You are shown a new advertisement:

Theme: {ad['theme']}
Color Tone: {ad['color_tone']}
Appeal Type: {ad['rhetorical_triangle']}
Text Amount: {ad['text_amount']}

Based on your personality and past behavior, how do you respond to this ad?

Available actions (select all that apply, following compatibility rules):
- ignore: You don't engage with it at all
- click: You click to learn more
- like: You like/favorite it
- dislike: You dislike/hide it
- share: You share it with others

Compatibility rules:
- If you ignore it, you can't share
- If you click, you can also like, dislike, or share
- If you like or dislike, you can share
- You must take at least one action

Respond in JSON format:
{{
  "ignore": true/false,
  "click": true/false,
  "like": true/false,
  "dislike": true/false,
  "share": true/false,
  "reasoning": "brief explanation"
}}"""

        # Call OpenAI API
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a realistic social media user responding to advertisements based on your personality and interests."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=200
        )

        # Parse response
        try:
            result = json.loads(response.choices[0].message.content)
            actions = {
                "ignore": result.get("ignore", False),
                "click": result.get("click", False),
                "like": result.get("like", False),
                "dislike": result.get("dislike", False),
                "share": result.get("share", False)
            }

            # Record interaction
            self.history.append({
                "ad": ad,
                "actions": actions,
                "reasoning": result.get("reasoning", "")
            })

            return actions
        except json.JSONDecodeError:
            # Fallback: default to ignore
            return {
                "ignore": True,
                "click": False,
                "like": False,
                "dislike": False,
                "share": False
            }
```

### 1.3 Run Baseline Simulation to Collect Data

```python
# world/simulator.py

import duckdb
from agents.agent import AdAgent
from typing import List, Dict
import random
import json
from datetime import datetime

class AdSimulator:
    def __init__(self, config_path="config/simulation_config.yaml"):
        # Load configuration
        import yaml
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        # Initialize database
        self.db = duckdb.connect("data/interactions.db")
        self._init_database()

        # Load agents and ads
        self.agents = self._load_agents()
        self.ads = self._load_ads()

        self.current_day = 0

    def _init_database(self):
        """Create interactions table"""
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS interactions (
                interaction_id INTEGER PRIMARY KEY,
                agent_id VARCHAR,
                ad_id VARCHAR,
                day INTEGER,
                timestamp TIMESTAMP,
                ignore BOOLEAN,
                click BOOLEAN,
                like BOOLEAN,
                dislike BOOLEAN,
                share BOOLEAN,
                agent_age VARCHAR,
                agent_gender VARCHAR,
                agent_hobbies VARCHAR,
                agent_activity_level FLOAT,
                agent_risk_tolerance FLOAT,
                agent_social_engagement FLOAT,
                ad_theme VARCHAR,
                ad_color_tone VARCHAR,
                ad_rhetorical_triangle VARCHAR,
                ad_text_amount VARCHAR
            )
        """)

    def _load_agents(self) -> List[AdAgent]:
        """Load agents from users.json"""
        with open("data/users.json") as f:
            user_data = json.load(f)

        agents = []
        for user in user_data:
            agent = AdAgent(
                agent_id=user['id'],
                features=user
            )
            agents.append(agent)

        return agents

    def _load_ads(self) -> List[Dict]:
        """Load ads from ads.json"""
        with open("data/ads.json") as f:
            return json.load(f)

    def run_day(self):
        """Run one simulation day"""
        self.current_day += 1
        print(f"\n=== Day {self.current_day} ===")

        # Select ads to show
        num_exposures = self.config['no_agents_exposed_to_ad']

        for ad in self.ads:
            # Randomly select agents
            selected_agents = random.sample(self.agents, min(num_exposures, len(self.agents)))

            for agent in selected_agents:
                # Get agent's decision
                actions = agent.decide_action_baseline(ad, temperature=self.config['decision_temperature'])

                # Log interaction
                self._log_interaction(agent, ad, actions)

        print(f"Logged {len(self.ads) * num_exposures} interactions")

    def _log_interaction(self, agent: AdAgent, ad: Dict, actions: Dict):
        """Log interaction to database"""
        self.db.execute("""
            INSERT INTO interactions VALUES (
                NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
        """, [
            agent.agent_id,
            ad['id'],
            self.current_day,
            datetime.now(),
            actions['ignore'],
            actions['click'],
            actions['like'],
            actions['dislike'],
            actions['share'],
            agent.age,
            agent.gender,
            ','.join(agent.hobbies),
            agent.activity_level,
            agent.risk_tolerance,
            agent.social_engagement,
            ad['theme'],
            ad['color_tone'],
            ad['rhetorical_triangle'],
            ad['text_amount']
        ])

    def export_training_data(self, output_path="data/tinker_training_data.jsonl"):
        """
        Export interactions as training data for Tinker fine-tuning
        """
        query = """
            SELECT
                agent_id,
                ad_id,
                ignore, click, like, dislike, share,
                agent_age, agent_gender, agent_hobbies,
                agent_activity_level, agent_risk_tolerance, agent_social_engagement,
                ad_theme, ad_color_tone, ad_rhetorical_triangle, ad_text_amount
            FROM interactions
            ORDER BY day, interaction_id
        """

        results = self.db.execute(query).fetchall()

        training_examples = []
        for row in results:
            # Reconstruct prompt
            prompt = f"""User Profile:
- Age: {row[7]}
- Gender: {row[8]}
- Hobbies: {row[9]}

Personality Traits:
- Activity Level: {row[10]}%
- Risk Tolerance: {row[11]}%
- Social Engagement: {row[12]}%

Advertisement:
- Theme: {row[13]}
- Color Tone: {row[14]}
- Appeal Type: {row[15]}
- Text Amount: {row[16]}

How do you respond to this ad? (JSON format with ignore/click/like/dislike/share fields)"""

            # Reconstruct response
            response = json.dumps({
                "ignore": row[2],
                "click": row[3],
                "like": row[4],
                "dislike": row[5],
                "share": row[6]
            })

            training_examples.append({
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response}
                ]
            })

        # Save to JSONL
        with open(output_path, "w") as f:
            for example in training_examples:
                f.write(json.dumps(example) + "\n")

        print(f"Exported {len(training_examples)} training examples to {output_path}")
        return output_path

# Usage
if __name__ == "__main__":
    sim = AdSimulator()

    # Run baseline simulation for 20 days to collect data
    for _ in range(20):
        sim.run_day()

    # Export training data for Tinker
    sim.export_training_data()
```

---

## Phase 2: Supervised Fine-Tuning with Tinker

### 2.1 Prepare Training Data

```python
# tinker/prepare_data.py

import json
import duckdb
from collections import defaultdict

def prepare_persona_specific_data(db_path="data/interactions.db"):
    """
    Split training data by user clusters (personas)
    for training separate LoRA adapters
    """
    db = duckdb.connect(db_path)

    # Cluster users (simplified - in practice use K-means)
    query = """
        SELECT DISTINCT
            agent_id,
            agent_hobbies,
            agent_activity_level,
            agent_risk_tolerance,
            agent_social_engagement
        FROM interactions
    """

    users = db.execute(query).fetchall()

    # Simple clustering based on hobbies (in practice: use sklearn K-means)
    persona_map = defaultdict(list)
    for user in users:
        hobbies = user[1].lower()

        if 'sport' in hobbies or 'fitness' in hobbies:
            persona = "sports_enthusiasts"
        elif 'tech' in hobbies or 'gaming' in hobbies:
            persona = "tech_enthusiasts"
        elif user[2] < 30:  # Low activity
            persona = "passive_users"
        elif user[4] > 70:  # High social engagement
            persona = "influencers"
        else:
            persona = "general_users"

        persona_map[persona].append(user[0])

    # Export data per persona
    for persona, agent_ids in persona_map.items():
        agent_list = ','.join([f"'{aid}'" for aid in agent_ids])

        query = f"""
            SELECT
                agent_id,
                ignore, click, like, dislike, share,
                agent_age, agent_gender, agent_hobbies,
                agent_activity_level, agent_risk_tolerance, agent_social_engagement,
                ad_theme, ad_color_tone, ad_rhetorical_triangle, ad_text_amount
            FROM interactions
            WHERE agent_id IN ({agent_list})
        """

        results = db.execute(query).fetchall()

        # Create training examples
        training_data = []
        for row in results:
            prompt = f"""You are a {persona.replace('_', ' ')} on social media.

Your Profile:
- Age: {row[6]}
- Gender: {row[7]}
- Hobbies: {row[8]}
- Activity Level: {row[9]}%
- Risk Tolerance: {row[10]}%
- Social Engagement: {row[11]}%

You see an ad:
- Theme: {row[12]}
- Color Tone: {row[13]}
- Appeal: {row[14]}
- Text Amount: {row[15]}

How do you respond? (JSON: ignore/click/like/dislike/share)"""

            response = json.dumps({
                "ignore": row[1],
                "click": row[2],
                "like": row[3],
                "dislike": row[4],
                "share": row[5]
            })

            training_data.append({
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response}
                ]
            })

        # Save persona-specific data
        output_file = f"data/tinker_{persona}_training.jsonl"
        with open(output_file, "w") as f:
            for example in training_data:
                f.write(json.dumps(example) + "\n")

        print(f"Created {len(training_data)} examples for {persona}")

if __name__ == "__main__":
    prepare_persona_specific_data()
```

### 2.2 Fine-Tune Base Model

```python
# tinker/train_base_model.py

import tinker
from tinker import AsyncClient
import json

async def train_ad_agent_model():
    """
    Fine-tune base model on all agent interactions
    """
    client = AsyncClient()

    # Load training data
    def load_training_data():
        with open("data/tinker_training_data.jsonl") as f:
            for line in f:
                yield json.loads(line)

    # Training configuration
    config = {
        "model": "qwen3-8b-instruct",  # Good multilingual support
        "lora_rank": 32,
        "lora_alpha": 64,
        "learning_rate": 2e-4,
        "batch_size": 8,
        "num_steps": 2000,  # Adjust based on data size
    }

    print(f"Training {config['model']} for ad agent behavior...")

    step = 0
    for batch in batched(load_training_data(), config['batch_size']):
        # Forward and backward pass
        loss = await client.forward_backward(
            messages=batch,
            loss_type="cross_entropy"
        )

        # Update weights
        await client.optim_step()

        step += 1

        if step % 100 == 0:
            print(f"Step {step}/{config['num_steps']}, Loss: {loss:.4f}")

            # Save checkpoint
            await client.save_state(f"models/ad_agent_step_{step}")

        if step >= config['num_steps']:
            break

    # Save final model
    await client.save_weights_for_sampler("models/ad_agent_base")
    print("Base model training complete!")

def batched(iterable, n):
    """Batch data into chunks"""
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == n:
            yield batch
            batch = []
    if batch:
        yield batch

if __name__ == "__main__":
    import asyncio
    asyncio.run(train_ad_agent_model())
```

### 2.3 Fine-Tune Persona-Specific LoRA Adapters

```python
# tinker/train_persona_adapters.py

import tinker
from tinker import AsyncClient
import json
import asyncio

async def train_persona_adapter(persona: str):
    """
    Train LoRA adapter for specific user persona
    """
    client = AsyncClient()

    # Load base model first
    await client.load_state("models/ad_agent_base")

    # Load persona-specific data
    data_file = f"data/tinker_{persona}_training.jsonl"

    def load_persona_data():
        with open(data_file) as f:
            for line in f:
                yield json.loads(line)

    print(f"\nTraining {persona} adapter...")

    step = 0
    for batch in batched(load_persona_data(), batch_size=4):
        loss = await client.forward_backward(
            messages=batch,
            loss_type="cross_entropy"
        )

        await client.optim_step()
        step += 1

        if step % 50 == 0:
            print(f"  {persona}: Step {step}, Loss: {loss:.4f}")

        if step >= 500:  # Fewer steps for adapter fine-tuning
            break

    # Save persona adapter
    await client.save_weights_for_sampler(f"models/adapter_{persona}")
    print(f"✓ {persona} adapter complete!")

async def train_all_personas():
    personas = [
        "sports_enthusiasts",
        "tech_enthusiasts",
        "passive_users",
        "influencers",
        "general_users"
    ]

    for persona in personas:
        await train_persona_adapter(persona)

def batched(iterable, n):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == n:
            yield batch
            batch = []
    if batch:
        yield batch

if __name__ == "__main__":
    asyncio.run(train_all_personas())
```

---

## Phase 3: RLVR for Behavioral Realism

### 3.1 Define Realism Reward Function

```python
# tinker/reward_function.py

import json
from typing import Dict

class RealismRewardFunction:
    """
    Reward function that encourages realistic, persona-consistent behavior
    """

    def __init__(self, persona: str):
        self.persona = persona

        # Define expected behavior patterns per persona
        self.persona_preferences = {
            "sports_enthusiasts": {
                "themes": ["sport", "fitness", "health"],
                "engagement_rate": 0.6,  # High engagement
                "share_rate": 0.4,
            },
            "tech_enthusiasts": {
                "themes": ["technology", "gaming", "innovation"],
                "engagement_rate": 0.7,
                "share_rate": 0.3,
            },
            "passive_users": {
                "themes": [],  # No strong preferences
                "engagement_rate": 0.2,  # Low engagement
                "share_rate": 0.05,
            },
            "influencers": {
                "themes": ["fashion", "lifestyle", "trending"],
                "engagement_rate": 0.8,
                "share_rate": 0.6,  # Very high sharing
            },
            "general_users": {
                "themes": [],
                "engagement_rate": 0.4,
                "share_rate": 0.2,
            }
        }

    def calculate_reward(self,
                        ad: Dict,
                        actions: Dict[str, bool],
                        agent_features: Dict) -> float:
        """
        Calculate reward based on behavioral realism

        Positive rewards for:
        - Persona-consistent theme engagement
        - Realistic engagement rates
        - Diversity in responses

        Negative rewards for:
        - Violating compatibility rules
        - Unrealistic always-ignore or always-engage patterns
        """
        reward = 0.0

        prefs = self.persona_preferences[self.persona]

        # 1. Theme consistency (+2 if matches persona interests)
        if ad['theme'] in prefs['themes']:
            if actions['click'] or actions['like'] or actions['share']:
                reward += 2.0
            elif actions['ignore']:
                reward -= 0.5  # Should engage with preferred themes

        # 2. Engagement rate consistency
        is_engaged = actions['click'] or actions['like'] or actions['share']

        if is_engaged:
            # Reward for engagement when expected
            reward += prefs['engagement_rate']
        else:
            # Reward for ignoring when expected (passive users)
            reward += (1 - prefs['engagement_rate'])

        # 3. Sharing behavior
        if actions['share']:
            # Reward appropriate to persona
            reward += prefs['share_rate'] * 2

        # 4. Action compatibility validation (penalty for violations)
        if not self._validate_compatibility(actions):
            reward -= 5.0  # Heavy penalty for impossible combinations

        # 5. Diversity bonus (avoid always same response)
        # (Would track history in real implementation)

        # 6. Dislike penalty (should be rare)
        if actions['dislike']:
            reward -= 0.3

        return reward

    def _validate_compatibility(self, actions: Dict[str, bool]) -> bool:
        """
        Check action compatibility rules
        """
        # ignore + share is incompatible
        if actions['ignore'] and actions['share']:
            return False

        # Must take at least one action
        if not any(actions.values()):
            return False

        # All other combinations are valid per spec
        return True
```

### 3.2 RLVR Training Loop

```python
# tinker/train_rlvr.py

import tinker
from tinker import AsyncClient
import json
import random
from tinker.reward_function import RealismRewardFunction

async def rlvr_train_persona(persona: str, num_episodes=500):
    """
    Use RLVR to fine-tune persona adapter for realistic behavior
    """
    client = AsyncClient()

    # Load persona adapter
    await client.load_state(f"models/adapter_{persona}")

    reward_fn = RealismRewardFunction(persona)

    # Load ads for environment
    with open("data/ads.json") as f:
        ads = json.load(f)

    # Training loop
    print(f"\nRLVR training for {persona}...")

    episode = 0
    batch_size = 16

    while episode < num_episodes:
        batch_trajectories = []
        batch_rewards = []

        for _ in range(batch_size):
            # Sample random ad
            ad = random.choice(ads)

            # Generate agent response
            prompt = f"""You are a {persona.replace('_', ' ')}.

Ad:
- Theme: {ad['theme']}
- Color Tone: {ad['color_tone']}
- Appeal: {ad['rhetorical_triangle']}
- Text: {ad['text_amount']}

Respond (JSON: ignore/click/like/dislike/share):"""

            response = await client.sample(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=50,
                temperature=0.7
            )

            # Parse actions
            try:
                actions = json.loads(response)
            except:
                # Fallback for parsing errors
                actions = {
                    "ignore": True,
                    "click": False,
                    "like": False,
                    "dislike": False,
                    "share": False
                }

            # Calculate reward
            reward = reward_fn.calculate_reward(
                ad=ad,
                actions=actions,
                agent_features={"persona": persona}
            )

            batch_trajectories.append({
                "prompt": prompt,
                "response": response,
                "ad": ad,
                "actions": actions
            })
            batch_rewards.append(reward)

        # Update model using policy gradient
        await client.forward_backward(
            trajectories=batch_trajectories,
            rewards=batch_rewards,
            loss_type="importance_sampling"
        )

        await client.optim_step()

        episode += batch_size

        avg_reward = sum(batch_rewards) / len(batch_rewards)
        print(f"  Episode {episode}/{num_episodes}, Avg Reward: {avg_reward:.3f}")

        # Save checkpoint
        if episode % 100 == 0:
            await client.save_state(f"models/rlvr_{persona}_ep{episode}")

    # Save final model
    await client.save_weights_for_sampler(f"models/rlvr_{persona}_final")
    print(f"✓ RLVR training complete for {persona}!")

async def train_all_personas_rlvr():
    personas = [
        "sports_enthusiasts",
        "tech_enthusiasts",
        "passive_users",
        "influencers",
        "general_users"
    ]

    for persona in personas:
        await rlvr_train_persona(persona, num_episodes=500)

if __name__ == "__main__":
    import asyncio
    asyncio.run(train_all_personas_rlvr())
```

---

## Phase 4: Production Integration

### 4.1 Tinker Agent Wrapper

```python
# agents/tinker_agent.py

import tinker
from tinker import AsyncClient
import json
from typing import Dict

class TinkerAdAgent:
    """
    Ad agent using Tinker fine-tuned model instead of API calls
    """

    def __init__(self, agent_id: str, features: Dict, persona: str):
        self.agent_id = agent_id
        self.persona = persona

        # Features (same as baseline)
        self.age = features.get("age")
        self.gender = features.get("gender")
        self.hobbies = features.get("hobbies", [])
        self.activity_level = features.get("activity_level", 50)
        self.risk_tolerance = features.get("risk_tolerance", 50)
        self.social_engagement = features.get("social_engagement", 50)

        # Tinker client (shared across agents)
        self.client = None
        self.model_loaded = False

    async def initialize(self, client: AsyncClient):
        """Initialize with shared Tinker client"""
        self.client = client

        if not self.model_loaded:
            # Load persona-specific model
            await self.client.load_weights(f"models/rlvr_{self.persona}_final")
            self.model_loaded = True

    async def decide_action(self, ad: Dict, temperature=0.5) -> Dict[str, bool]:
        """
        Use Tinker model to decide action (replaces API call)

        ~0.1-0.5 seconds vs 1-3 seconds for API
        ~$0.0001 vs $0.001+ for API call
        """
        prompt = f"""You are a {self.persona.replace('_', ' ')}.

Your Profile:
- Age: {self.age}
- Hobbies: {', '.join(self.hobbies)}
- Activity: {self.activity_level}%
- Risk Tolerance: {self.risk_tolerance}%
- Social: {self.social_engagement}%

Ad:
- Theme: {ad['theme']}
- Color: {ad['color_tone']}
- Appeal: {ad['rhetorical_triangle']}
- Text: {ad['text_amount']}

Respond (JSON: ignore/click/like/dislike/share):"""

        response = await self.client.sample(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50,
            temperature=temperature
        )

        # Parse JSON response
        try:
            actions = json.loads(response.strip())

            # Validate and return
            return {
                "ignore": actions.get("ignore", False),
                "click": actions.get("click", False),
                "like": actions.get("like", False),
                "dislike": actions.get("dislike", False),
                "share": actions.get("share", False)
            }
        except json.JSONDecodeError:
            # Fallback
            return {
                "ignore": True,
                "click": False,
                "like": False,
                "dislike": False,
                "share": False
            }
```

### 4.2 Production Simulator with Tinker

```python
# world/tinker_simulator.py

from agents.tinker_agent import TinkerAdAgent
from tinker import AsyncClient
import asyncio
import duckdb
import json

class TinkerAdSimulator:
    """
    Production simulator using Tinker models
    """

    def __init__(self, config_path="config/simulation_config.yaml"):
        import yaml
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.db = duckdb.connect("data/interactions_tinker.db")
        self._init_database()

        # Tinker client (shared)
        self.client = AsyncClient()

        self.agents = []
        self.ads = []
        self.current_day = 0

    def _init_database(self):
        """Same as baseline"""
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS interactions (
                interaction_id INTEGER PRIMARY KEY,
                agent_id VARCHAR,
                ad_id VARCHAR,
                day INTEGER,
                timestamp TIMESTAMP,
                ignore BOOLEAN,
                click BOOLEAN,
                like BOOLEAN,
                dislike BOOLEAN,
                share BOOLEAN,
                persona VARCHAR,
                ad_theme VARCHAR
            )
        """)

    async def initialize(self):
        """Load agents and ads"""
        # Load agents with persona assignments
        with open("data/users_with_personas.json") as f:
            users = json.load(f)

        for user in users:
            agent = TinkerAdAgent(
                agent_id=user['id'],
                features=user,
                persona=user['persona']
            )
            await agent.initialize(self.client)
            self.agents.append(agent)

        # Load ads
        with open("data/ads.json") as f:
            self.ads = json.load(f)

        print(f"Initialized {len(self.agents)} agents, {len(self.ads)} ads")

    async def run_day(self):
        """Run one simulation day"""
        self.current_day += 1
        print(f"\n=== Day {self.current_day} ===")

        num_exposures = self.config['no_agents_exposed_to_ad']

        for ad in self.ads:
            # Select agents
            import random
            selected = random.sample(self.agents, min(num_exposures, len(self.agents)))

            # Get decisions concurrently
            tasks = [agent.decide_action(ad) for agent in selected]
            action_results = await asyncio.gather(*tasks)

            # Log interactions
            for agent, actions in zip(selected, action_results):
                self._log_interaction(agent, ad, actions)

        print(f"Completed {len(self.ads) * num_exposures} interactions")

    def _log_interaction(self, agent, ad, actions):
        """Log to database"""
        from datetime import datetime

        self.db.execute("""
            INSERT INTO interactions VALUES (
                NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
        """, [
            agent.agent_id,
            ad['id'],
            self.current_day,
            datetime.now(),
            actions['ignore'],
            actions['click'],
            actions['like'],
            actions['dislike'],
            actions['share'],
            agent.persona,
            ad['theme']
        ])

async def main():
    """Run full simulation"""
    sim = TinkerAdSimulator()
    await sim.initialize()

    # Run for 100 days
    for _ in range(100):
        await sim.run_day()

    print("\n✓ Simulation complete!")

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Evaluation & Comparison

### Benchmark Script

```python
# evaluation/compare_baseline_vs_tinker.py

import duckdb
import matplotlib.pyplot as plt
import pandas as pd

def compare_results():
    """
    Compare baseline (GPT-4-mini) vs Tinker results
    """

    # Load both databases
    db_baseline = duckdb.connect("data/interactions.db")
    db_tinker = duckdb.connect("data/interactions_tinker.db")

    # Compare engagement rates
    query = """
        SELECT
            day,
            AVG(CAST(click AS FLOAT)) as click_rate,
            AVG(CAST(like AS FLOAT)) as like_rate,
            AVG(CAST(share AS FLOAT)) as share_rate,
            AVG(CAST(ignore AS FLOAT)) as ignore_rate
        FROM interactions
        GROUP BY day
        ORDER BY day
    """

    df_baseline = db_baseline.execute(query).df()
    df_tinker = db_tinker.execute(query).df()

    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    metrics = ['click_rate', 'like_rate', 'share_rate', 'ignore_rate']
    titles = ['Click Rate', 'Like Rate', 'Share Rate', 'Ignore Rate']

    for ax, metric, title in zip(axes.flat, metrics, titles):
        ax.plot(df_baseline['day'], df_baseline[metric], label='Baseline (GPT-4-mini)', linewidth=2)
        ax.plot(df_tinker['day'], df_tinker[metric], label='Tinker (Fine-tuned)', linewidth=2, linestyle='--')
        ax.set_xlabel('Day')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(title)
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('evaluation/baseline_vs_tinker.png', dpi=300)
    print("✓ Saved comparison plot")

    # Cost comparison
    baseline_cost = len(df_baseline) * 100 * 10 * 0.0005  # agents * days * exposures * cost per call
    tinker_cost = len(df_tinker) * 100 * 10 * 0.0001

    print(f"\n=== Cost Comparison ===")
    print(f"Baseline (GPT-4-mini): ${baseline_cost:.2f}")
    print(f"Tinker (Fine-tuned):   ${tinker_cost:.2f}")
    print(f"Savings: {(1 - tinker_cost/baseline_cost)*100:.1f}%")

if __name__ == "__main__":
    compare_results()
```

---

## Expected Results

### Performance Metrics

| Metric | Baseline (GPT-4-mini) | Tinker (Fine-tuned) |
|--------|----------------------|---------------------|
| **Cost (100 days)** | $25-50 | $0.50-1 (98% reduction) |
| **Latency per decision** | 1-3 seconds | 0.1-0.5 seconds |
| **Total simulation time** | 5-10 hours | 30-60 minutes |
| **Behavioral realism** | General LLM patterns | Persona-specific, trained |
| **Scalability** | Limited by rate limits | 1000+ agents easily |
| **Engagement variance** | Lower (consistent API) | Higher (realistic diversity) |

### Timeline

- **Week 1**: Implement baseline, run 20-day simulation, collect data
- **Week 2**: Fine-tune base model and persona adapters with Tinker
- **Week 3**: RLVR training for behavioral realism
- **Week 4**: Production integration, evaluation, comparison

---

## Next Steps

1. **Implement baseline simulation** (Phase 1) to generate training data
2. **Set up Tinker account** and obtain API key
3. **Export training data** from DuckDB interactions
4. **Run supervised fine-tuning** (Phase 2)
5. **Train persona-specific adapters** for specialized behavior
6. **Implement RLVR** (Phase 3) for realism rewards
7. **Integrate into production simulator**
8. **Benchmark and compare** vs baseline

---

## Advanced: Multi-Objective RLVR

For research extensions, you can use multi-objective rewards:

```python
# Advanced reward combining multiple objectives
reward = (
    0.4 * realism_score +        # Behavioral consistency
    0.3 * diversity_score +       # Avoid repetitive responses
    0.2 * engagement_quality +    # Meaningful interactions
    0.1 * compatibility_bonus     # Valid action combinations
)
```

This enables:
- More nuanced agent behaviors
- Research into optimal reward weighting
- Publication-worthy ablation studies

---

## Resources

- **Tinker Docs**: https://tinker-docs.thinkingmachines.ai/
- **DuckDB Docs**: https://duckdb.org/docs/
- **Project Spec**: `docs/specifications.md`
- **Tinker Integration Analysis**: `/projects/TINKER_INTEGRATION_ANALYSIS.md`

---

*Implementation Guide v1.0*
*Last Updated: 2025-11-13*
*Project: Social Media Ads Simulation*
