# Human vs AI Flappy Bird gameplay analysis

## Introduction
The goal of this project is to use reinforcement learning algorithms (e.g. Q-learning, Deep Q-Network) or evolutionary/genetic algorithms to train an agent to play a Flappy Bird–style game autonomously. 
The goal is to teach the model to maximize its score by learning optimal actions based on the game’s state (bird position, pipe distance, velocity, etc.). 
Human and agent performance during playtime would then be extensively explored and compared for purposes of analysis.

**Optional extensions include:**
- Evaluating performance of language models (e.g., Gemini, GPT) as simplified agents.
- Modifying the game’s difficulty and observing how model performance adapts. 

## Hypothesis
Humans will adapt quicker in the Flappy bird clone game, but if given enough time, AI agents will (in a significant amount of cases) have a greater score than a human ever could.

## Technologies
For the "human" analysis part and agent performance visualization, the plan is to use and repurpose the  existing Flappy Bird clone recreated in JavaScript.
However, the agents (such as reinforcement learning agents) would not be trained here but in a Flappy Bird clone "engine". This Python engine would not have a GUI like the JS version, but would make an ideal model training platform.
This engine is planned to be created in a Google Colab notebook so all needed datasets and variables would be accessible easily during the "analysis" part of this project.

## Data collection
The necessary datasets would be created primarily from agent and human gameplay variable states such as bird position, pipe distance, velocity, etc.

## Diagram
![Diagram](human-vs-ai-fb-play-analysis-diagram.svg)

## Link to project
https://github.com/Rei0101/human-vs-ai-fb-play-analysis
