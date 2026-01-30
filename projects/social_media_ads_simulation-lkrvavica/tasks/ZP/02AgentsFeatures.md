# Agent Features extraction

- Extracting agent features from dataset.

### Objectives and research question

Process the RedDust dataset to extract personal features and generate propensity features that will participate in the simulation.

### Detailed specs

#### Functional requirements
- Extract personal features from RedDust dataset
- Generate random propensity features (0-100%)

#### Technical requirements
- Data processing pipeline for RedDust dataset
- Random number generation with seed control

#### Subtasks
1. **Extract features from dataset** - Process RedDust dataset to extract personal features
- fetch data using Zendo endpoint: https://zenodo.org/api/records/3541657 
- create a dataframe that assigns each row with user_id, combines features, how to combine them is explained in dataset's README
2. **EDA analysis** - do a quick EDA on dataframe
- remove any missing rows
- visualise distributions and correlations
3. **Choose agents** - choose which lines of dataframe to keep 
- decide which agents_count number of lines to keep to maintain as much diversity as posible by analysing the visualisations above
- save dataframe in world_sim/data/users_features.csv

#### Dependencies

#### Input data
- RedDust dataset
- Simulation config (number of agents, seed)

#### Output
- `user_feature_extraction/user_feature_extraction.py` - user feature extraction script 
- `data/users_features.csv` - agent descriptions

## Workflow, algorithms and procedures
1.  **Data Fetching**: The script starts by fetching metadata from the Zenodo API endpoint for the RedDust dataset. It extracts the download links for the raw data files.
2.  **Data Loading**: The script downloads the raw text files containing user features like age, gender, profession, hobby, and family status.
3.  **Data Processing**: The script then processes these files to create a single aggregated dataset. It samples 1000 users and creates a dataframe with one row per user, containing all their features. 
4.  **Exploratory Data Analysis (EDA)**: The script performs an EDA on the cleaned dataset. 
    - It visualizes the distribution of numerical features (age) using histograms to understand the age spread of the users.
    - It visualizes the distribution of categorical features (gender, profession, hobby, family) using count plots to see the composition of the user base.
    - It removes any rows with missing data to ensure the quality of the dataset.
    - It removes lines with hobbies and professions esstimated to be invalid
5.  **Agent Selection**: All users with multiple hobbies are selected because they carry more information. The rest users are choosen randomly, but each with different proffesion.
6.  **Data filling**: Although not the best practice, data is modified and each user is given 3 to 7 new hobbies.

## Issues and challenges
- The dataset documentation did not contain clear instructions on how to combine features. For this task, we assumed a one-to-one mapping between rows in `dataset.csv` and individual users.
- Adding almost random hobbies to users is definitely not the best practice. The better solution would be to find a new dataset with better user descriptions or create the new dataset that fits this project.

## Results and conclusions
- A Marimo-based Python script (`user_feature_extraction/user_feature_extraction.py`) was created to automate the process of fetching, cleaning, analyzing, and selecting user features.
- An EDA was performed, revealing the basic statistical properties and distributions of the user features. The analysis shows a diverse range of ages, professions, and hobbies. For example, the age distribution is skewed towards younger users, and there is a wide variety of professions and hobbies.
- A dataset of 1000 users with diverse features was generated and saved to `data/users_features.csv`, ready to be used in the simulation. After cleaning, all 1000 users were kept as they had complete data, providing a solid foundation for the agent-based modeling part of the project.

## Notes
The agent selection process can be refined in the future to select a subset of agents based on specific criteria, such as selecting a certain number of agents from each profession or age group to create a more controlled simulation environment.