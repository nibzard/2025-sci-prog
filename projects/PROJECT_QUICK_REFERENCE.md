# Scientific Programming Course - Projects Quick Reference

## Project Overview Table

| # | Project Name | Student | Domain | ML Type | LLM | RL | Status |
|---|---|---|---|---|---|---|---|
| 1 | Location Prediction | Marija Karoglan | Transportation | Regression | No | No | In Progress |
| 2 | LLM Chess Battle | Nikola Vidović | Gaming/AI | LLM Agents | Yes* | No | In Progress |
| 3 | Property Price Estimator | Maksimilijan Katavić | Real Estate | Regression + LLM | Yes | No | In Progress |
| 4 | Facial Emotion Recognition | Srđan Machiedo | Computer Vision | CNN | No | No | In Progress |
| 5 | Intrusion Detection System | Mariela Uvodic | Cybersecurity | Classification | No | No | In Progress |
| 6 | Shipping Delay Prediction | Petra Krišto | Logistics | Classification | No | No | In Progress |
| 7 | Weather Impact Analysis | Pavao Katavić | Environmental | Regression | No | No | In Progress |
| 8 | Flight Delay Prediction | Marul Babic | Transportation | Regression | No | No | In Progress |
| 9 | Used Cars Price Prediction | Antonio Jurjevic | E-commerce | Regression | No | No | In Progress |
| 10 | AI Flappy Bird | Rei Krstić | Gaming | RL + Optional LLM | Optional | Yes* | In Progress |
| 11 | SiteSecGym | Paško Stipandžija | Cybersecurity/Testing | Agent Testing | Yes* | Yes* | In Progress |
| 12 | Social Media Ads Simulation | Lara Krvavica | Marketing | Agent-Based + Clustering | Yes | No | In Progress |
| 13 | Portfolio Intelligence Dashboard | Juraj Mestrović | Finance | Sentiment + RAG | Yes | No | In Progress |
| 14 | LaptopCompare AI | Marko Rados | E-commerce | Sentiment Analysis | Yes | No | In Progress |
| 15 | Voltage Outlier Detection | Maja Nakić | Utilities | Anomaly Detection | No | No | In Progress |
| 16 | Biomass & Biodiversity Analysis | Marin Jovanović | Marine Science | Statistical Analysis | No | No | In Progress |

*Core component

---

## Technical Stack by Category

### Languages & Frameworks
- **Python**: 15/16 projects (dominant)
- **JavaScript**: 1 (Flappy Bird GUI)

### ML Libraries
- **scikit-learn**: Classification, clustering, anomaly detection
- **TensorFlow/Keras**: CNN (emotion recognition), DQN (Flappy Bird)
- **PyTorch**: Optional for DQN
- **XGBoost/RandomForest**: Regression and classification

### Data Processing
- **Pandas**: Data manipulation (13+ projects)
- **NumPy**: Numerical operations
- **Matplotlib/Seaborn/Plotly**: Visualization

### Database & Storage
- **PostgreSQL**: Structured data (3 projects)
- **ChromaDB**: Vector embeddings and semantic search (2 projects)
- **DuckDB**: Lightweight analytics (1 project)

### LLM APIs
- **OpenAI GPT-4/GPT-4-mini**: Chess, social ads
- **Anthropic Claude**: Chess agents
- **Google Gemini**: Portfolio dashboard, embeddings
- **Groq**: Sentiment analysis (laptop comparison)

### Web Scraping & Automation
- **BeautifulSoup**: HTML parsing
- **Selenium**: Browser automation
- **Puppeteer/Playwright**: Advanced browser automation
- **Exa AI**: Semantic web search
- **PRAW**: Reddit API
- **APIs**: Meteostat, Alpha Vantage, ActuallyFreeAPI, aqicn.org

### Web Frameworks
- **FastAPI**: Backend (portfolio, potential others)
- **React**: Frontend (portfolio)
- **Flask**: Optional for others

---

## Project Complexity Ranking

### Most Complex (Full Stack)
1. **Portfolio Intelligence Dashboard** - Multiple APIs, real-time updates, RAG, full stack
2. **LaptopCompare AI** - Multiple data sources, NLP, vector DB, complex scoring
3. **Social Media Ads Simulation** - Agent-based modeling, ML, database

### Advanced (Multi-component ML)
4. **Property Price Estimator** - Web scraping, LLM, feature engineering, multiple models
5. **SiteSecGym** - Agent testing framework, multiple agent types, risk quantification

### Intermediate (Standard ML Pipeline)
6. **Shipping Delay Prediction** - Classification, EDA, model comparison
7. **Flight Delay Prediction** - Regression, feature analysis
8. **Weather Impact Analysis** - Data integration, regression, visualization
9. **AI Flappy Bird** - RL algorithms, environment simulation

### Standard (Well-defined ML Problem)
10. **Facial Emotion Recognition** - CNN classification, standard dataset
11. **Intrusion Detection System** - Classification on security data
12. **Used Cars Price Prediction** - Regression, web scraping
13. **Location Prediction** - Regression with domain knowledge
14. **LLM Chess Battle** - LLM integration, game logic

### Exploratory (Data Analysis)
15. **Voltage Outlier Detection** - Anomaly detection, simpler scope
16. **Biomass & Biodiversity** - Statistical analysis, correlation studies

---

## Data Sources Summary

### APIs
- **Alpha Vantage**: Stock prices
- **Meteostat**: Weather data
- **ActuallyFreeAPI**: News articles
- **aqicn.org**: Air quality
- **Exa AI**: Web search (semantic)

### Kaggle Datasets
- FER2013: Emotion recognition
- Supply Chain Shipment: Shipping delays
- Airline Delay: Flight predictions
- Shipment data: Logistics

### Web Scraping Targets
- Real estate portals (Croatia)
- mobile.de, AutoScout24 (cars)
- GSMArena, NotebookCheck (specs)
- Reddit (r/laptops, etc.)
- YouTube (transcripts)
- Booking.com, Airbnb (tourism, optional)

### Research Data
- Institute for Oceanography & Fisheries (biomass)
- Public boating incident reports (location prediction)
- Electrical meter databases (voltage anomalies)

---

## Feature Engineering Approaches

### Common Techniques
1. **Temporal features**: Day of week, time of day, season
2. **Categorical encoding**: OneHot, LabelEncoding
3. **Normalization**: Min-max, standardization
4. **Domain-specific features**:
   - Distance calculations (Walk Score, proximity)
   - Aggregations (average, sum, count)
   - Ratios (shipment_value = cost/weight)
   - Embeddings (from LLMs or pre-trained models)

---

## Evaluation Metrics Used

### Classification
- Accuracy, Precision, Recall, F1-score
- ROC-AUC, Confusion Matrix
- Precision-Recall curves

### Regression
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² (Coefficient of Determination)
- MAPE (Mean Absolute Percentage Error)

### Clustering
- Silhouette score
- Davies-Bouldin index
- Inertia

### Custom Metrics
- **Risk score** (SiteSecGym): 0-1 scale, weighted validator deltas
- **Interaction rate** (Social Ads): Weighted action formula
- **Engagement metrics**: CTR, like rate, share rate

---

## Challenges & Solutions

### Common Challenges
| Challenge | Projects | Solution |
|---|---|---|
| Class imbalance | Emotion, Intrusion | Over/under-sampling, class weights |
| Missing data | Weather, Properties | Imputation, feature selection |
| Location uncertainty | Properties | Centroid approximation, confidence radius |
| Non-representative data | Emotion (FER2013) | Augmentation, domain adaptation |
| Real-time data volume | Portfolio | Caching, batch processing |
| Model validation | Most | Cross-validation, hold-out test sets |

---

## Deployment & Infrastructure

### Completed
- Full-stack web app (Portfolio Dashboard)
- Web sandbox environment (SiteSecGym)

### Planned
- REST APIs (Property estimator, potentially others)
- Web interfaces (Multiple projects)
- Docker containerization (General practice)

---

## Research Questions by Project

1. **Location Prediction**: Can ocean currents + environmental factors predict boat location?
2. **Chess Battle**: How do different LLMs compare in strategic game-playing?
3. **Property Estimator**: Can LLM-extracted features improve property price predictions?
4. **Emotion Recognition**: Can CNNs accurately classify emotions from 48x48 images?
5. **Intrusion Detection**: Can ML classify network attacks from TCP packet features?
6. **Shipping Delays**: Which logistics factors most influence delivery delays?
7. **Weather Impact**: Do weather patterns predict air quality or tourism patterns?
8. **Flight Delays**: Can historical data predict future flight delays accurately?
9. **Car Prices**: How well can regression models predict used car prices?
10. **Flappy Bird AI**: Can RL agents exceed human performance in game playing?
11. **SiteSecGym**: How vulnerable are different agent types to web-based threats?
12. **Social Ads**: Can clustering + ML improve ad targeting over random distribution?
13. **Portfolio Dashboard**: Does RAG improve investment decision-making quality?
14. **LaptopCompare**: Does multi-source scoring outperform spec-only comparisons?
15. **Voltage Anomalies**: Can outlier detection identify problematic meter readings?
16. **Marine Biomass**: How does fishing pressure affect ecosystem biodiversity?

---

## Innovation Highlights

### Novel Approaches
- **Multi-source scoring** (LaptopCompare): Combines specs, community, expert opinions
- **LLM feature extraction** (Property estimator): Using LLMs for data enrichment
- **Agent-based modeling** (Social ads): Simulating individual user behaviors
- **Web threat testing** (SiteSecGym): Systematically testing agent robustness
- **RAG for finance** (Portfolio): Context-aware investment advice

### Emerging Technologies
- Vector databases (ChromaDB) for semantic search
- LLM agents as decision-makers (chess, ads, testing)
- Exa AI for semantic web search
- Groq for fast inference

---

## Course Learning Outcomes Demonstrated

### Scientific Programming Skills
- Data processing and cleaning
- Statistical analysis
- Algorithm implementation
- Software engineering practices (Git, structure)

### ML/AI Skills
- Supervised learning (regression, classification)
- Unsupervised learning (clustering, anomaly detection)
- Deep learning (CNNs, DQNs)
- Reinforcement learning
- LLM integration and prompt engineering
- Agent-based modeling

### Data Science Skills
- Web scraping and API integration
- Exploratory data analysis
- Feature engineering
- Model evaluation and comparison
- Visualization and interpretation

### Production Skills
- Full-stack development (some projects)
- Database design
- API development
- Data pipeline construction

