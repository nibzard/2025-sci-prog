# Scientific Programming Course - Project Directory Analysis

## Overview
This comprehensive analysis covers 15 student projects from the Scientific Programming course. Projects span diverse domains including machine learning, reinforcement learning, web scraping, data analysis, and AI agent development.

---

## PROJECT SUMMARIES

### 1. Location Prediction - Marija Karoglan
**Status**: In Progress

**Main Objective**:
- Create a predictive model for the location of a boat after it sends an SOS signal
- Consider various environmental variables (currents, time, etc.) when predicting location

**Current Approach/Methodology**:
- Data sourced from public reports of boating incidents
- Use scholastic articles to understand ocean current formulas and impacts
- Mathematical equations balancing different metrics

**ML/AI Components**:
- Predictive modeling using environmental variables
- Curve fitting or regression for location prediction

**LLM Usage**: None mentioned

**RL Components**: None

---

### 2. LLM Agents Chess Battle - Nikola Vidović
**Status**: In Progress

**Main Objective**:
- Implement an automated chess playing system using LLM agents
- Have agents compete across three competitive disciplines

**Current Approach/Methodology**:
- Use LLM APIs (OpenAI GPT, Anthropic Claude, local models) to power chess-playing agents
- Integrate with Steel browser for web-based interaction and gameplay
- Use python-chess library for game logic and board manipulation
- Selenium for additional browser automation

**Technologies**:
- Python 3.8+, Steel Browser, LLM APIs
- python-chess, Selenium, Pandas & NumPy for analysis
- Matplotlib & Seaborn for visualization

**LLM Usage**: 
- Core component: LLM agents make chess moves
- OpenAI GPT, Anthropic Claude, and local models as decision-making engines

**ML/AI Components**:
- LLM-based agent decision making
- Performance analysis and statistics

**RL Components**: None

---

### 3. Croatian Property Price Estimator - Maksimilijan Katavić
**Status**: In Progress

**Main Objective**:
- Build the first Croatian real estate price estimation tool (inspired by Zillow Zestimate)
- Provide reliable market value predictions based on comprehensive data analysis

**Current Approach/Methodology**:
- **Phase 1**: Web scraping from multiple real estate portals; data cleaning and validation
- **Phase 2**: Exploratory data analysis; feature engineering; train multiple ML models
- **Phase 3**: Deploy backend API and web interface with continuous model updates

**Key Features**:
- Web scraping for automated data collection
- **LLM-powered feature extraction** from unstructured property descriptions
- Geolocation analysis and proximity calculations (Walk Score, nearby amenities)
- Price prediction using property characteristics, location factors, extracted features, and market trends

**ML/AI Components**:
- Regression models for price prediction
- Feature engineering and geolocation analysis
- LLM for text feature extraction from property descriptions

**LLM Usage**:
- Core component: Extract additional features from free-text property descriptions
- Improve feature set for ML models

**RL Components**: None

---

### 4. Facial Emotion Recognition - Srđan Machiedo
**Status**: In Progress

**Main Objective**:
- Develop a system to recognize emotions on human faces from images
- Classify seven emotions: anger, disgust, fear, happiness, sadness, surprise, neutral

**Current Approach/Methodology**:
1. Data loading: FER2013 dataset (35,000 grayscale 48×48 pixel images)
2. Data preprocessing: Pixel normalization (0-1), one-hot encoding
3. Model training: Convolutional Neural Network (CNN)
4. Evaluation: Accuracy/loss curves, confusion matrices
5. Prediction: Test on unseen images; optional real-time recognition via OpenCV

**ML/AI Components**:
- Convolutional Neural Networks (CNN) for image classification
- Deep learning for emotion classification

**LLM Usage**: None

**RL Components**: None

---

### 5. Intrusion Detection System - Mariela Uvodic
**Status**: In Progress

**Main Objective**:
- Detect and classify cyber attacks by analyzing TCP packet content
- Build an AI-powered system for network security

**Current Approach/Methodology**:
- Analyze incoming network traffic (TCP packets)
- Use AI models to classify traffic as malicious or benign
- Identify specific attack types
- Use pre-existing security datasets with packet features

**ML/AI Components**:
- Classification models (specific type to be determined)
- Analyze TCP packet features for attack detection

**LLM Usage**: None

**RL Components**: None

---

### 6. Shipping Delay Prediction - Petra Krišto
**Status**: In Progress

**Main Objective**:
- Develop ML model to predict shipment delivery delays
- Help logistics companies optimize operations and reduce delay rates

**Current Approach/Methodology**:
1. Data preparation: Supply Chain Shipment Dataset (10,000+ records)
2. Data cleaning and categorical encoding
3. Exploratory Data Analysis (EDA) with visualization
4. Feature engineering: is_heavy, discount_category, shipment_value
5. Model training: Compare Logistic Regression, Decision Tree, Random Forest
6. Evaluation: Accuracy, Precision, Recall, F1-score, ROC-AUC

**ML/AI Components**:
- Multiple classification algorithms
- Feature engineering and EDA
- Performance comparison and visualization

**LLM Usage**: None

**RL Components**: None

---

### 7. Weather Impact Analysis - Pavao Katavić
**Status**: In Progress

**Main Objective**:
- Explore relationships between weather conditions and external factors
- Test if weather (temperature, humidity, wind, pressure) predicts air quality or tourism

**Current Approach/Methodology**:
1. **Data scraping**: Weather from Meteostat API; Air quality from aqicn.org or weather stations
2. **Data integration**: Merge datasets by date and location
3. **Analysis**: Correlation analysis between weather and target variables
4. **Visualization**: Matplotlib, Plotly (scatter plots, heatmaps, time series)
5. **Modeling**: RandomForestRegressor, XGBoost
6. **Evaluation**: R², MAE, RMSE metrics

**ML/AI Components**:
- Regression models (RandomForest, XGBoost)
- Data preprocessing and correlation analysis

**LLM Usage**: None

**RL Components**: None

---

### 8. Flight Delay Prediction - Marul Babic
**Status**: In Progress

**Main Objective**:
- Predict flight delays using historical data and flight conditions
- Help airlines optimize operations

**Current Approach/Methodology**:
- Use historical flight data from Kaggle dataset
- Analyze impact of multiple factors: departure/arrival airports, scheduled time, airline, day of week, weather
- Train linear and logistic regression models

**ML/AI Components**:
- Regression and logistic regression models
- Analysis of feature importance

**LLM Usage**: None

**RL Components**: None

---

### 9. Used Cars Price Prediction - Antonio Jurjevic
**Status**: In Progress

**Main Objective**:
- Predict used car prices based on vehicle characteristics
- Help buyers and sellers make data-driven decisions

**Current Approach/Methodology**:
- Web scraping from mobile.de and AutoScout24
- Features: brand, model, year, mileage, fuel_type, transmission, power_hp, price
- Machine learning regression to predict prices

**ML/AI Components**:
- Regression models for price prediction
- Web scraping and data collection

**LLM Usage**: None

**RL Components**: None

---

### 10. AI Flappy Bird Analysis - Rei Krstić
**Status**: In Progress

**Main Objective**:
- Train AI agents to autonomously play Flappy Bird game
- Compare agent vs human gameplay behavior extensively
- Test LLM performance as simplified agents (optional)

**Current Approach/Methodology**:
- JavaScript-based Flappy Bird clone for human gameplay and visualization
- Python-based Flappy Bird engine (Google Colab notebook) for agent training
- Agent types: Reinforcement Learning (Q-learning, Deep Q-Network) or Genetic Algorithms
- Data collection: Bird position, pipe distance, velocity, etc.
- Optional: Difficulty modification to observe adaptation

**ML/AI Components**:
- Deep Q-Networks (DQN)
- Genetic/Evolutionary algorithms
- Game state analysis

**Reinforcement Learning Components** (Core):
- Q-learning algorithms
- Deep Q-Networks (DQN)
- Policy learning for optimal decision-making

**LLM Usage** (Optional):
- Evaluate LLM performance (Gemini, GPT) as simplified agents

---

### 11. SiteSecGym - Paško Stipandžija
**Status**: In Progress

**Main Objective**:
- Create a controlled web sandbox for testing LLM agents, browser automation agents, and RL agents
- Evaluate agent behavior against malicious and risky web elements
- Detect vulnerabilities: prompt injections, fake forms, data collection scripts

**Current Approach/Methodology**:
- Expose agents to static and dynamic web threats
- Collect telemetry: clicks, form submissions, network requests
- Automatic evaluation: Pass/Warn/Fail verdicts from validators
- Risk scoring: Quantify agent vulnerability (0-1 scale)
- Generate dataset for agent training and safety evaluation

**ML/AI Components**:
- Agent behavior analysis
- Risk scoring and classification

**LLM Usage**:
- Core component: Test LLM agents against web-based threats
- Evaluate how LLMs handle prompt injection and malicious inputs

**RL Components**:
- RL agents tested against web threats
- Behavioral analysis of RL-based web agents

---

### 12. Social Media Ads Simulation - Lara Krvavica
**Status**: In Progress

**Main Objective**:
- Simulate how users interact with online advertisements
- Optimize ad exposure across user groups to maximize engagement
- Build a realistic dataset mimicking social media ad dynamics

**Current Approach/Methodology**:
1. **Agent-based modeling**: Each user represented as an autonomous agent with personality traits and behavioral tendencies
2. **Advertisement representation**: Structured ads with theme, color tone, rhetorical triangle, text amount
3. **Simulation**: Agents react to ads (like, click, ignore, share, dislike)
4. **Data collection**: Log all interactions
5. **ML model training**: Predict how user groups respond to ads
6. **User clustering**: K-Means, DBSCAN, hierarchical clustering for segmentation

**ML/AI Components**:
- Agent-based modeling
- User segmentation/clustering (K-Means, DBSCAN)
- Supervised learning for behavior prediction (Logistic Regression, Decision Trees, Random Forest)
- LLM agents for user behavioral simulation

**LLM Usage**:
- LLM-based agents make interaction decisions
- GPT-4-mini specified in configuration
- Agents respond to ads based on descriptions and history

**RL Components**: None

---

### 13. Portfolio Intelligence Dashboard - Juraj Mestrović
**Status**: In Progress

**Main Objective**:
- Provide comprehensive investment portfolio analysis with intelligent insights
- Integrate stock data, news sentiment, and AI-powered advisory

**Current Approach/Methodology**:
- **Data sources**: Alpha Vantage API (stock prices), ActuallyFreeAPI (news), company info
- **Architecture**: Full-stack with React frontend, FastAPI backend, PostgreSQL + ChromaDB
- **Components**:
  - Real-time stock price tracking (18 stocks across multiple sectors)
  - News aggregation and sentiment analysis
  - Vector embeddings for semantic search (ChromaDB)
  - AI-powered chatbot using Retrieval-Augmented Generation (RAG)

**ML/AI Components**:
- News sentiment analysis
- Vector embeddings for semantic search
- Natural language processing for chatbot queries

**LLM Usage**:
- Google Gemini API: Text generation, embeddings, NLP for chatbot
- RAG (Retrieval-Augmented Generation) for contextual answers
- AI chatbot for investment questions using market data and news context

**Data Processing**:
- Data validation and quality checks
- Real-time WebSocket updates
- 324,000+ stock price records
- 1000+ news articles with embeddings

---

### 14. LaptopCompare AI - Marko Rados
**Status**: In Progress

**Main Objective**:
- Create comprehensive laptop comparison tool combining specs with real-world user feedback
- Synthesize objective specifications, Reddit community feedback, and YouTube review analysis

**Current Approach/Methodology**:
1. **Data sources**:
   - **Technical specs**: Manufacturer sites, GSMArena, NotebookCheck (via Puppeteer/Playwright)
   - **Reddit feedback**: Exa AI search for r/laptops, r/SuggestALaptop, etc.; sentiment analysis
   - **YouTube reviews**: Extract transcripts; analyze reviewer sentiment and test results
2. **Scoring**: Three independent 0-100 scores (Specs, Reddit sentiment, YouTube reviews)
3. **AI analysis**: Groq API for NLP sentiment analysis
4. **Vector database**: ChromaDB for semantic search and SEO optimization
5. **Visualization**: Radar charts comparing three dimensions

**ML/AI Components**:
- Sentiment analysis (Reddit comments, YouTube transcripts)
- Text analysis and feature extraction
- Unified scoring system combining multiple data sources

**LLM Usage**:
- Groq API: Sentiment analysis and scoring of Reddit posts and YouTube transcripts
- Exa AI: Semantic search for relevant Reddit posts and YouTube videos
- NLP for extracting pros/cons from unstructured review text

**Technologies**:
- ChromaDB (vector database), PostgreSQL
- Groq API, Exa AI, PRAW (Reddit API)
- YouTube Transcript API

---

### 15. Traženje outliera među mjerenjima napona - Maja Nakić
**Status**: In Progress

**Main Objective**:
- Detect anomalous voltage measurements in three-phase electrical meter readings
- Identify when device maintenance or corrections are needed

**Current Approach/Methodology**:
- Dataset: Three-phase meter measurements at single time points
- Challenge: Some meters measure through voltage transformers (58V expected) vs direct (230V expected)
- Use outlier detection techniques to distinguish normal vs anomalous measurements
- Send technicians to problematic locations for investigation and fixes

**ML/AI Components**:
- Outlier detection algorithms
- Anomaly detection in numerical data

**Technologies**:
- Python, Pandas, scikit-learn

**LLM Usage**: None

**RL Components**: None

---

### 16. Analiza biomase i bioraznolikosti - Marin Jovanović
**Status**: In Progress

**Main Objective**:
- Analyze biomass and biodiversity in fishing areas with different fishing pressure
- Compare allowed vs forbidden fishing zones
- Understand ecological processes and fish population dynamics

**Current Approach/Methodology**:
1. **Data source**: Institute for Oceanography and Fisheries
2. **Data**: Commercial fishing survey data with catch records, species counts, depth, location, zone status
3. **Analysis**:
   - Compare biomass between allowed/forbidden zones
   - Analyze ecosystem structure and stability
   - Correlation analysis: depth vs species presence; species interactions
   - Size analysis: fish length vs catch zone
4. **Modeling**: Population reproduction and sustainability models
5. **Visualization**: Graphs, correlation networks, spatial maps

**ML/AI Components**:
- Statistical analysis and correlation analysis
- Possible population dynamics modeling
- Data visualization and pattern recognition

**Technologies**:
- Python, Pandas, Matplotlib, statistical methods

**LLM Usage**: None

**RL Components**: None

---

## SUMMARY STATISTICS

### Total Projects: 16

### ML/AI Component Breakdown:
- **Pure ML Projects**: 11
  - Prediction/regression: 6 (flights, shipping, cars, properties, weather, location)
  - Classification: 4 (emotion recognition, intrusion detection, anomaly detection, biomass analysis)
  - Clustering/segmentation: 1 (social media ads)

- **LLM-powered Projects**: 4
  - Chess agents (LLM-based decision making)
  - Property estimator (LLM feature extraction)
  - Portfolio dashboard (RAG chatbot)
  - Laptop comparison (LLM sentiment analysis)

- **RL Projects**: 2
  - Flappy Bird (Q-learning, DQN, Genetic algorithms)
  - SiteSecGym (RL agents + LLM agents testing)

- **Agent-based Modeling**: 2
  - Social media ads simulation
  - SiteSecGym

- **Web Scraping + ML**: 5
  - Property estimator
  - Laptop comparison
  - Weather impact
  - Used cars
  - LaptopCompare

### Domain Distribution:
- **Finance/Investment**: 3 (portfolio, property, cars)
- **Transportation/Logistics**: 3 (flights, shipping, boats)
- **Security/Safety**: 2 (intrusion detection, SiteSecGym)
- **Environmental/Marine**: 2 (weather, biomass)
- **Computer Vision/NLP**: 2 (emotion recognition, sentiment analysis)
- **Gaming/Agents**: 1 (Flappy Bird)
- **Marketing**: 1 (social media ads)
- **Electronics/Utilities**: 1 (voltage anomaly)
- **AI/LLM Testing**: 1 (chess agents)
- **Data Analysis**: 1 (laptop comparison, biomass)

### Technology Trends:
- **Python**: Primary language for 15/16 projects
- **scikit-learn**: Used in multiple projects
- **LLM APIs**: GPT-4, Claude, Gemini, Groq
- **Web scraping**: BeautifulSoup, Selenium, Playwright, Puppeteer
- **Databases**: PostgreSQL, ChromaDB, DuckDB
- **Deep Learning**: TensorFlow/PyTorch (emotion recognition, DQN)
- **Web frameworks**: FastAPI, React

### Key Findings:
1. **Strong ML focus**: Nearly all projects use supervised or unsupervised learning
2. **Growing LLM integration**: 4 projects use LLMs as core components, not just tools
3. **Data-driven approach**: 14/16 projects rely on external data (APIs, scraping, datasets)
4. **Real-world applications**: Projects address practical problems (property pricing, flight delays, network security)
5. **Diverse RL usage**: Flappy Bird uses classical RL; SiteSecGym tests RL agents in web environment
6. **Agent-based modeling**: Emerging trend with social ads and SiteSecGym projects

