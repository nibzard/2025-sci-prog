# Model Mocker Creation

- Creation of lightweight mock models and placeholder logic to enable early integration and testing before full model availability.

## Objectives and research question

- **Objective:** Enable system development and integration by providing mock implementations of core models and lifecycle hooks.  
- **Research question:** How can simplified mock models best simulate expected behavior while keeping implementation minimal and flexible?

## Detailed specs

### Functional requirements
- Provide a mock version of the Target Public Prediction Model (TPPM).  
- Implement a simple and fast user grouping model based on K-means clustering.  
- Define a placeholder function triggered when the model is updated.  
- Persist mock models to storage for reuse across sessions.  

### Technical requirements
- Programming language: Python.  
- ML libraries: scikit-learn (for K-means), NumPy, pandas.  
- File storage: local filesystem or object storage.  

### Subtasks
1. **Simple grouping model** - Implement a lightweight K-means-based user grouping model
- Create a K-means grouping algorithm, based only on personal, propensity and frend_list user features(without history), that will group the users only once
- save it in DuckDB in grouping table
2. **TPPM mocker** - Implement a mock version of the Target Public Prediction Model (TPPM)
- Create a function that will randomly assign each agent with a number that will represent the model's prediction of should the ad be shown to user
3. **Prediction query handler** - Create a main function that will recieve an ad description, run the TPPM mocker model for each user and calculate the average for each group of users, scale it so that the sum of each group is equal to agents_exposed_to_ad, so that it represents the number of people.

### Dependencies
- Defined interfaces for TPPM and grouping models.  
- Agreement on expected input/output formats for mocked components.  

### Input data

### Output
- `world\initialization.py` - simple grouping model
- `world\simulator.py` - target public prediction mocker, prediction query handler
## Workflow, algorithms and procedures
1. **Simple Grouping Model (Subtask 1):**
   - A K-means clustering algorithm is implemented to group users based on a selection of their features.
   - The features used for clustering include personal attributes (gender, age, profession, hobby, family), propensity features (activity level, risk tolerance, social engagement), and the number of friends.
   - Categorical features are one-hot encoded to be used in the K-means algorithm.
   - The number of clusters is set to 5.
   - The resulting user groups are saved to the `users_grouping` table in the DuckDB database for future use.

2. **Prediction Query Handler (Subtask 3):**
   - A main function, `prediction_query_handler`, is created to simulate the targeting process.
   - It takes an ad description as input and uses a mock Target Public Prediction Model (TPPM).
   - The mock TPPM assigns a random prediction score to each user.
   - The handler calculates the average prediction score for each user group (as determined by the grouping model).
   - The group averages are then scaled so that their sum equals the `agents_exposed_to_ad` value from the simulation configuration, ensuring the results represent a realistic distribution of ad exposure.

## Issues and challenges
- the real model will not be available, so mocking model output is the easiest solution
## Results and conclusions
- The `simple_grouping_model` successfully groups users into 5 clusters based on their features and stores the groupings in the database. This enables the system to perform group-based analysis and targeting.
- The `prediction_query_handler` provides a functional placeholder for the TPPM, allowing the simulation to proceed with ad targeting logic. It demonstrates how to integrate the grouping model with the prediction model and scale the results according to the simulation parameters.
- These mock implementations enable parallel development and testing of other system components that depend on the modeling and targeting functionalities, without having to wait for the final models to be developed.

