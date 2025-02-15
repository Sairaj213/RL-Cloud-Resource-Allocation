# RL-Cloud-Resource-Allocation Project

This project explores the application of Reinforcement Learning (RL) for dynamically managing cloud computing resources. The goal is to optimize performance, energy efficiency, and cost by intelligently managing resources such as CPU, memory, and GPU across multiple servers. The project includes a custom RL agent to make intelligent decisions based on the current state of the cloud environment.

## Project Structure

```markdown-tree
rl_cloud_resource_allocation/
├── environment.py           # Simulates the cloud environment
├── server.py                # Manages server resources
├── task.py                  # Represents workloads or tasks
├── rl_agent.py              # Implements the RL algorithm
├── task_generator.py        # Generates tasks with different patterns
├── cloud_simulation.py      # Manages the overall cloud simulation logic
├── app.py                   # Streamlit interface
├── requirements.txt         # Requirement
└── README.md                # Project documentation
```

## Description

- `environment.py` - Simulates the cloud environment, managing servers and tasks.
- `server.py` - Manages server resources and allocates them to tasks.
- `task.py` - Represents workloads with specific resource requirements and durations.
- `rl_agent.py` - Implements the reinforcement learning algorithm for dynamic resource allocation.
- `task_generator.py` - Generates tasks with different patterns (uniform, bursty, pattern-based).
- `cloud_simulation.py` - Manages the overall cloud simulation logic.
- `app.py` - Provides a user-friendly Streamlit interface for running and visualizing the simulation.
- `README.md` - Documentation for the project.

## Setup and Usage

### 1. Clone the Repository
```bash
https://github.com/Sairaj213/RL-Cloud-Resource-Allocation.git
cd rl_cloud_resource_allocation
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
Ensure `requirements.txt` includes necessary libraries such as `streamlit`, `plotly`, `pandas`, `numpy`, and `matplotlib`.

### 3. Run the Streamlit App
Run the Streamlit app to configure and start the simulation.
```bash
streamlit run app.py
```
The app allows you to configure the simulation settings, visualize resource usage in real-time, and observe the RL agent's performance.

## Simulation Settings

Configure the simulation settings using the Streamlit sidebar:

- **Max Time Steps**: Number of time steps to run the simulation.
- **Server Configurations**: Set the number of servers and their CPU, memory, and GPU configurations.
- **Task Generator**: Choose from uniform, bursty, or pattern-based task generation.
- **RL Agent Parameters**: Adjust the learning rate, discount factor, and exploration parameters.
- **Model Management**: Specify the model filename and optionally reset the model.

## Visualization

The main screen of the Streamlit app provides real-time visualizations of:

- **Server Resource Usage**: Interactive plots showing CPU, memory, and GPU utilization for each server.
- **Tasks Over Time**: Line plot of pending and completed tasks over time.
- **Rewards Over Time**: Line plot of the rewards received by the RL agent.

## Customization

- **Advanced Algorithms**: Modify `rl_agent.py` to implement advanced RL algorithms such as Deep Q-Networks (DQNs) or Policy Gradient Methods.
- **Data Augmentation**: Enhance `task_generator.py` by adding more task generation patterns to simulate different workload scenarios.
- **Hyperparameters**: Adjust parameters such as learning rate, discount factor, and exploration decay in `rl_agent.py`.
- **Streamlit Interface**: Customize `app.py` to modify the Streamlit interface.

## Model Persistence

The RL agent's model is saved to a file (`rl_agent_model.pkl`) after each simulation run and loaded at the start to ensure continuous learning. Use the Streamlit sidebar to specify the model filename or reset the model if needed.

