

import numpy as np
import random
import pickle
import os

class RLAgent:
    def __init__(self, action_size, learning_rate=0.1, discount_factor=0.9,
                 exploration_rate=1.0, exploration_decay=0.995, exploration_min=0.01,
                 model_filename='rl_agent_model.pkl'):
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = discount_factor

        
        self.epsilon = exploration_rate
        self.epsilon_decay = exploration_decay
        self.epsilon_min = exploration_min

        
        self.q_table = {}  

        
        self.model_filename = model_filename
        self.load_model()

    def get_state(self, servers, task):
        

        server_states = []
        for server in servers:
            util = server.get_utilization()
            server_states.extend([
                round(util['cpu'], 1),
                round(util['memory'], 1),
                round(util['gpu'], 1)
            ])
        task_state = [
            task.cpu_requirement,
            task.memory_requirement,
            task.gpu_requirement
        ]
        state = tuple(server_states + task_state)
        return state

    def select_action(self, servers, task):
        

        state = self.get_state(servers, task)

        
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_size)

        
        if random.uniform(0, 1) < self.epsilon:
            
            action = random.randint(0, self.action_size - 1)
        else:
            
            action = int(np.argmax(self.q_table[state]))

        return action

    def learn(self, prev_state, action, reward, next_state):
        

        
        if prev_state not in self.q_table:
            self.q_table[prev_state] = np.zeros(self.action_size)
        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(self.action_size)

        
        old_value = self.q_table[prev_state][action]
        future_optimal_value = np.max(self.q_table[next_state])

        
        td_target = reward + self.gamma * future_optimal_value
        td_error = td_target - old_value

        
        self.q_table[prev_state][action] += self.learning_rate * td_error

        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_parameters(self, learning_rate=None, discount_factor=None,
                          exploration_rate=None, exploration_decay=None, exploration_min=None):
        

        if learning_rate is not None:
            self.learning_rate = learning_rate
        if discount_factor is not None:
            self.gamma = discount_factor
        if exploration_rate is not None:
            self.epsilon = exploration_rate
        if exploration_decay is not None:
            self.epsilon_decay = exploration_decay
        if exploration_min is not None:
            self.epsilon_min = exploration_min

    def save_model(self):
        with open(self.model_filename, 'wb') as f:
            pickle.dump({
                'q_table': self.q_table,
                'epsilon': self.epsilon
            }, f)
        print(f"Model saved to {self.model_filename}")

    def load_model(self):
        if os.path.exists(self.model_filename):
            with open(self.model_filename, 'rb') as f:
                data = pickle.load(f)
                self.q_table = data.get('q_table', {})
                self.epsilon = data.get('epsilon', self.epsilon)
            print(f"Model loaded from {self.model_filename}")
        else:
            print("No saved model found. Starting with a fresh agent.")
