# World Definition & Ad Creation <br><div style="display:flex; justify-content:space-between; align-items:center;">  <span style="font-size:16px;">Status: Not started</span></div>

Defining and creating the world, creating ad templates, and generating ads.

### Subtasks
<table style="width:100%; border-collapse:collapse;">
  <tr><th style="width:80%; text-align:left;">Subtask</th><th style="width:20%; text-align:left;">Status</th></tr>
  <tr><td>Create world boundaries and specifications</td><td>Not started</td></tr>
  <tr><td>Create a function that determines which ads will be shown each day</td><td>Not started</td></tr>
  <tr><td>Create ad templates</td><td>Not started</td></tr>
  <tr><td>Create the ads</td><td>Not started</td></tr>
</table>

# Interaction History Management System <br><div style="display:flex; justify-content:space-between; align-items:center;">  <span style="font-size:16px;">Status: Not started</span></div>

Creating a system to store interaction history and determine the optimal agent data compression method.

### Subtasks
<table style="width:100%; border-collapse:collapse;">
  <tr><th style="width:80%; text-align:left;">Subtask</th><th style="width:20%; text-align:left;">Status</th></tr>
  <tr><td>Create an interaction history management system that stores each interaction</td><td>Not started</td></tr>
  <tr><td>Create multiple schemes for agent history data compression</td><td>Not started</td></tr>
  <tr><td>Select the best compression system</td><td>Not started</td></tr>
  <tr><td>Refine the final design</td><td>Not started</td></tr>
</table>

# Agent Template Creation <br> <div style="display:flex; justify-content:space-between; align-items:center;">    <span style="font-size:16px;">Status: Not started</span> </div>

Creating a template for agents that describes their features, allows access to history, and defines possible actions.

### Subtasks
<table style="width:100%; border-collapse:collapse;">
  <tr><th style="width:80%; text-align:left;">Subtask</th><th style="width:20%; text-align:left;">Status</th></tr>
  <tr><td>Define agent features</td><td>Not started</td></tr>
  <tr><td>Create a function allowing agents to access their history</td><td>Not started</td></tr>
  <tr><td>Define agent actions</td><td>Not started</td></tr>
</table>

# Agent Creation <br><div style="display:flex; justify-content:space-between; align-items:center;">  <span style="font-size:16px;">Status: Not started</span></div>

Extracting agent features and creating agents.

### Subtasks
<table style="width:100%; border-collapse:collapse;">
  <tr><th style="width:80%; text-align:left;">Subtask</th><th style="width:20%; text-align:left;">Status</th></tr>
  <tr><td>Extract features from dataset</td><td>Not started</td></tr>
  <tr><td>Generate random features</td><td>Not started</td></tr>
  <tr><td>Create agents using the agent template</td><td>Not started</td></tr>
  <tr><td>Create a model to group agents</td><td>Not started</td></tr>
</table>

# Agent Prompting Handler <br> <div style="display:flex; justify-content:space-between; align-items:center;"> <span style="font-size:16px;">Status: Not started</span></div>

Deciding which agents to show the ad to and sending prompts.

### Subtasks
<table style="width:100%; border-collapse:collapse;">
  <tr><th style="width:80%; text-align:left;">Subtask</th><th style="width:20%; text-align:left;">Status</th></tr>
  <tr><td>Create a function to query the model</td><td>Not started</td></tr>
  <tr><td>Create a function to randomly select agents for the ad</td><td>Not started</td></tr>
  <tr><td>Create a function to format and send prompts to selected agents</td><td>Not started</td></tr>
</table>

# Response Handler <br> <div style="display:flex; justify-content:space-between; align-items:center;">   <span style="font-size:16px;">Status: Not started</span></div>

Store interaction information, handle sharing, and trigger model updates.

### Subtasks
<table style="width:100%; border-collapse:collapse;">
  <tr><th style="width:80%; text-align:left;">Subtask</th><th style="width:20%; text-align:left;">Status</th></tr>
  <tr><td>Create a function to:
<ul>
<li>Store interaction history</li>
<li>Call agent prompting if the response involves sharing</li>
</ul></td><td>Not started</td></tr>
  <tr><td>Create an event listener to prompt model updates at the end of each day</td><td>Not started</td></tr>
</table>