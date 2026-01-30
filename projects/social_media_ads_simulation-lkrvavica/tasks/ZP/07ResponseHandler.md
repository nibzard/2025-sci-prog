# Response Handler

- Store interaction information, handle sharing actions.

## Objectives and research question

Process agent responses from LLM, validate and store interactions, handle share propagation to other agents.

## Detailed specs

### Functional requirements
- Parse LLM responses 
- Validate action compatibility
- Store interactions in database
- Handle share action (propagate to another agent)
- Deactivate ads below threshold

### Technical requirements
- Python functions in `world/simulator.py` and `world/logger.py`
- Database write operations
- Interaction rate calculation: `[(click + share + 2·like) - (2·dislike + ignore)] / N`

### Subtasks
1. **Create a function to store interaction history and call agent prompting when sharing occurs** - Handle response processing and share propagation
- create a function that stores the interaction in database with respect to database's structure
- create a function that additionaly queries the user to interact with the ad if the ad was shared with them
2. **Users emotional state update** - create a function that updates user's emotional state
- Create and call the function that uses User class method to update user's emotional state
3. **Day simulation loop** - Create a loop that will call all the necessary functions for sending and processing the prompt
- create a batch of code that:
    - checks if ads_scheduled_for_day in context\simulation_state.yaml is empty, if it is, checks which day it is in context\simulation_state.yaml and calls the ad scheduling function, and save it's output in context\simulation_state.yaml in ads_scheduled_for_day, if it is not empty, than it takes the first ad from the list and calls the ad processing function

    - processing function, or more functions, depending on the complexity of the simulation and the visualisation requirements:
        - checks if ad is active, if not, exits the function
        - selects target agents using prediction logic
        - enters the loop for each agent:
            - creates and sends the promt
            - process the result
            - stores interaction in database
            - updates agent's emotional state 
            - checks and handles the share action
        - updates the ad's interaction rate
        - deactivates ad if below threshold
        - removes the ad from the list. 
4. **visualisation** - create a function that visualises the simulation
- create a function that visualises the simulation
    - a streamlit app, that has frames in this order:
    - first frame enables the selection of llm apis and models (instead of existing dropdowns), has the information which day it is and which ad's turn it is, and a button to start the simulation
    - second batch of frames shows the simulation's progress, shows all states, visualises every decision made by the system, and shows the simulation's progress
    - third frame shows the final results of the simulation and has the button that can call another loop of simulation
   

### Dependencies
- ...

### Input data
-...

### Output
- Response processing functions in `world/simulator.py`

## Workflow, algorithms and procedures

## Issues and challenges

## Results and conclusions

## Notes
- ads shown per day can vary, and agents shown per day also