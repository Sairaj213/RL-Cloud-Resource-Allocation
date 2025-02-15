

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import os

from environment import Environment
from rl_agent import RLAgent
import task_generator

def main():
    st.title("Reinforcement Learning for Dynamic Resource Allocation in Cloud Computing")

    st.sidebar.header("Simulation Settings")

    
    max_time_steps = st.sidebar.number_input("Max Time Steps", min_value=10, max_value=1000, value=100, step=10)

    
    st.sidebar.subheader("Server Configurations")
    num_servers = st.sidebar.number_input("Number of Servers", min_value=1, max_value=10, value=4)

    server_configs = []
    for i in range(num_servers):
        st.sidebar.markdown(f"**Server {i+1} Configuration**")
        with st.sidebar.expander(f"Configure Server {i+1}", expanded=False):
            cpu_cores = st.number_input(f"CPU Cores (Server {i+1})", min_value=1, max_value=64, value=8, key=f'cpu_{i}')
            memory_gb = st.number_input(f"Memory (GB) (Server {i+1})", min_value=1, max_value=256, value=32, key=f'mem_{i}')
            gpu_cores = st.number_input(f"GPU Cores (Server {i+1})", min_value=0, max_value=16, value=0, key=f'gpu_{i}')
            server_configs.append({'cpu_cores': cpu_cores, 'memory_gb': memory_gb, 'gpu_cores': gpu_cores})

    
    st.sidebar.subheader("Task Generator")
    task_gen_option = st.sidebar.selectbox("Select Task Generator", ("Uniform", "Bursty", "Pattern-Based"))

    
    if task_gen_option == "Uniform":
        num_tasks = st.sidebar.number_input("Tasks per Time Step", min_value=1, max_value=20, value=5)
        task_gen = lambda time_step: task_generator.generate_tasks_uniform(time_step, num_tasks=num_tasks)
    elif task_gen_option == "Bursty":
        burst_probability = st.sidebar.slider("Burst Probability", min_value=0.0, max_value=1.0, value=0.1)
        max_burst_size = st.sidebar.number_input("Max Burst Size", min_value=5, max_value=50, value=20)
        task_gen = lambda time_step: task_generator.generate_tasks_bursty(
            time_step, burst_probability=burst_probability, max_burst_size=max_burst_size)
    elif task_gen_option == "Pattern-Based":
        task_gen = task_generator.generate_tasks_with_pattern

    
    st.sidebar.subheader("RL Agent Parameters")
    learning_rate = st.sidebar.slider("Learning Rate", min_value=0.01, max_value=1.0, value=0.1)
    discount_factor = st.sidebar.slider("Discount Factor", min_value=0.1, max_value=1.0, value=0.9)
    exploration_rate = st.sidebar.slider("Initial Exploration Rate", min_value=0.0, max_value=1.0, value=1.0)
    exploration_decay = st.sidebar.slider("Exploration Decay", min_value=0.9, max_value=1.0, value=0.995)
    exploration_min = st.sidebar.slider("Minimum Exploration Rate", min_value=0.0, max_value=1.0, value=0.01)

    st.sidebar.subheader("Model Management")
    model_filename = st.sidebar.text_input("Model Filename", value="rl_agent_model.pkl")
    reset_model = st.sidebar.checkbox("Reset Model", value=False)
    if reset_model and os.path.exists(model_filename):
        os.remove(model_filename)
        st.sidebar.write(f"Model '{model_filename}' has been reset.")

    if st.sidebar.button("Run Simulation"):
        st.write("Starting Simulation...")

        
        rl_agent = RLAgent(
            action_size=num_servers,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            exploration_rate=exploration_rate,
            exploration_decay=exploration_decay,
            exploration_min=exploration_min,
            model_filename=model_filename  
        )

        
        env = Environment(
            num_servers=num_servers,
            server_configs=server_configs,
            task_generator=task_gen,
            rl_agent=rl_agent,
            max_time_steps=max_time_steps
        )

        
        server_data = {i: {'cpu': [], 'memory': [], 'gpu': []} for i in range(num_servers)}
        time_steps = []
        pending_tasks_list = []
        completed_tasks_list = []
        rewards = []

        
        progress_bar = st.progress(0)
        for t in range(max_time_steps):
            env.step()

            
            time_steps.append(env.time_step)
            pending_tasks = len(env.pending_tasks)
            completed_tasks = len(env.completed_tasks)
            pending_tasks_list.append(pending_tasks)
            completed_tasks_list.append(completed_tasks)
            rewards.append(env.log[-1]['reward'])

            
            for i, server in enumerate(env.servers):
                utilization = server.get_utilization()
                server_data[i]['cpu'].append(utilization['cpu'])
                server_data[i]['memory'].append(utilization['memory'])
                server_data[i]['gpu'].append(utilization['gpu'])

            
            progress_bar.progress((t + 1) / max_time_steps)

        
        rl_agent.save_model()

        st.success("Simulation completed and model saved.")

        
        st.subheader("Simulation Results")

        
        st.subheader("Server Resource Usage")
        for i in range(num_servers):
            df_server = pd.DataFrame({
                'Time Step': time_steps,
                'CPU Utilization': server_data[i]['cpu'],
                'Memory Utilization': server_data[i]['memory'],
                'GPU Utilization': server_data[i]['gpu'],
            })

            fig = px.line(df_server, x='Time Step',
                          y=['CPU Utilization', 'Memory Utilization', 'GPU Utilization'],
                          title=f'Server {i+1} Resource Utilization',
                          labels={'value': 'Utilization (%)', 'variable': 'Resource'},
                          hover_data={'Time Step': True, 'value': True, 'variable': True})
            st.plotly_chart(fig)

        
        st.subheader("Tasks Over Time")
        df_tasks = pd.DataFrame({
            'Time Step': time_steps,
            'Pending Tasks': pending_tasks_list,
            'Completed Tasks': completed_tasks_list,
        })
        fig_tasks = px.line(df_tasks, x='Time Step',
                            y=['Pending Tasks', 'Completed Tasks'],
                            labels={'value': 'Number of Tasks', 'variable': 'Task Status'},
                            title='Pending and Completed Tasks Over Time')
        st.plotly_chart(fig_tasks)

        
        st.subheader("Rewards Over Time")
        df_rewards = pd.DataFrame({
            'Time Step': time_steps,
            'Reward': rewards,
        })
        fig_rewards = px.line(df_rewards, x='Time Step', y='Reward', title='Reward Over Time')
        st.plotly_chart(fig_rewards)

    else:
        st.write("Adjust the simulation settings and click 'Run Simulation' to start.")

if __name__ == "__main__":
    main()
