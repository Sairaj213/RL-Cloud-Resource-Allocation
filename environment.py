

import random
from server import Server
from task import Task

class Environment:
    def __init__(self, num_servers, server_configs, task_generator, rl_agent, max_time_steps=100):
        self.time_step = 0
        self.max_time_steps = max_time_steps

        
        self.servers = []
        for i in range(num_servers):
            config = server_configs[i % len(server_configs)]
            server = Server(
                server_id=i,
                cpu_cores=config.get('cpu_cores', 8),
                memory_gb=config.get('memory_gb', 32),
                gpu_cores=config.get('gpu_cores', 0)
            )
            self.servers.append(server)

        self.task_generator = task_generator
        self.rl_agent = rl_agent

        self.pending_tasks = []
        self.completed_tasks = []
        self.log = []

    def reset(self):
        
        self.time_step = 0
        for server in self.servers:
            server.tasks = []
            server.available_cpu_cores = server.total_cpu_cores
            server.available_memory_gb = server.total_memory_gb
            server.available_gpu_cores = server.total_gpu_cores

        self.pending_tasks = []
        self.completed_tasks = []
        self.log = []

    def step(self):
    
        self.time_step += 1

        
        new_tasks = self.task_generator(self.time_step)
        self.pending_tasks.extend(new_tasks)

        
        for task in self.pending_tasks[:]:
            prev_state = self.rl_agent.get_state(self.servers, task)
            action = self.rl_agent.select_action(self.servers, task)
            server = self.servers[action]

            
            allocated = server.allocate_resources(task)
            if allocated:
                self.pending_tasks.remove(task)
                

                
                self.update_tasks()

                
                next_state = self.rl_agent.get_state(self.servers, task)

                
                reward = self.calculate_reward(task_completed=False)

                
                self.rl_agent.learn(prev_state, action, reward, next_state)
            else:
                
                reward = self.calculate_reward(allocation_failed=True)

                
                next_state = prev_state  
                self.rl_agent.learn(prev_state, action, reward, next_state)

        
        self.update_tasks()

        
        state = self.get_state()
        self.log.append({
            'time_step': self.time_step,
            'state': state,
            'reward': reward
        })

    def run(self):
    
        self.reset()
        while self.time_step < self.max_time_steps:
            self.step()

    def update_tasks(self):
       
        for server in self.servers:
            for task in server.tasks[:]:
                completed = task.step()
                if completed:
                    server.release_resources(task)
                    self.completed_tasks.append(task)

                    
                    reward = self.calculate_reward(task_completed=True)

                    
                    
                    state = self.rl_agent.get_state(self.servers, task)
                    action = server.server_id
                    next_state = state  
                    self.rl_agent.learn(state, action, reward, next_state)

    def get_state(self):
    
        state = {
            'time_step': self.time_step,
            'server_utilizations': [server.get_utilization() for server in self.servers],
            'num_pending_tasks': len(self.pending_tasks),
            'num_completed_tasks': len(self.completed_tasks)
        }
        return state

    def calculate_reward(self, task_completed=False, allocation_failed=False):

        reward = 0
        if task_completed:
            
            reward += 10
        if allocation_failed:
            
            reward -= 5

        
        
        reward -= len(self.pending_tasks) * 0.1

        return reward

    def get_logs(self):
        return self.log
