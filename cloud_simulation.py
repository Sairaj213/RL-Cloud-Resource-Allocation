import random
import numpy as np

class Server:
    def __init__(self, server_id, cpu_capacity, memory_capacity):
        self.server_id = server_id
        self.cpu_capacity = cpu_capacity
        self.memory_capacity = memory_capacity
        self.current_cpu_usage = 0
        self.current_memory_usage = 0
        self.running_applications = []

    def allocate_resources(self, application):
        if (self.current_cpu_usage + application.cpu_request <= self.cpu_capacity and
            self.current_memory_usage + application.memory_request <= self.memory_capacity):
            self.current_cpu_usage += application.cpu_request
            self.current_memory_usage += application.memory_request
            self.running_applications.append(application)
            return True
        else:
            return False

    def release_resources(self, application):
        if application in self.running_applications:
            self.current_cpu_usage -= application.cpu_request
            self.current_memory_usage -= application.memory_request
            self.running_applications.remove(application)
        
        

    def get_available_cpu(self):
    
        return self.cpu_capacity - self.current_cpu_usage

    def get_available_memory(self):
    
        return self.memory_capacity - self.current_memory_usage

    def get_current_utilization(self):
        cpu_utilization = (self.current_cpu_usage / self.cpu_capacity) * 100 if self.cpu_capacity > 0 else 0
        memory_utilization = (self.current_memory_usage / self.memory_capacity) * 100 if self.memory_capacity > 0 else 0
        return cpu_utilization, memory_utilization


class Application:
    def __init__(self, application_id, cpu_request, memory_request, duration, arrival_time):
        self.application_id = application_id
        self.cpu_request = cpu_request
        self.memory_request = memory_request
        self.duration = duration
        self.arrival_time = arrival_time
        self.status = "Pending"
        self.server_assigned_to = None
        self.start_time = None
        self.completion_time = None
        self.remaining_duration = duration
        self.wait_time = 0 

    def run(self, server, current_time):
        if server.allocate_resources(self):
            self.status = "Running"
            self.server_assigned_to = server.server_id
            self.start_time = current_time
            return True
        else:
            return False

    def update_status(self, current_time):
        if self.status == "Running":
            self.remaining_duration -= 1
            if self.remaining_duration <= 0:
                self.status = "Completed"
                self.completion_time = current_time
        elif self.status == "Pending": 
            self.wait_time += 1


    def complete(self, cloud_environment):
        if self.status == "Completed":
            if self.server_assigned_to and cloud_environment: 
                server = cloud_environment.get_server(self.server_assigned_to)
                if server:
                    server.release_resources(self)
                self.server_assigned_to = None

    def __str__(self):
        return (f"Application ID: {self.application_id}, Status: {self.status}, "
                f"CPU: {self.cpu_request}, Mem: {self.memory_request}, Dur: {self.duration}, Arr: {self.arrival_time}, "
                f"Server: {self.server_assigned_to}, Start: {self.start_time}, Comp: {self.completion_time}, RemDur: {self.remaining_duration}, WaitTime: {self.wait_time}")



class CloudEnvironment:
    def __init__(self, server_configs, arrival_rate, cpu_request_distribution, memory_request_distribution, duration_distribution,
                 reward_type='utilization_balancing', scheduling_policy='first_fit'): 
        self.servers = {}
        self.applications = []
        self.pending_applications = []
        self.running_applications = []
        self.completed_applications = []
        self.current_time = 0
        self.application_counter = 0
        self.arrival_rate = arrival_rate
        self.cpu_request_distribution = cpu_request_distribution
        self.memory_request_distribution = memory_request_distribution
        self.duration_distribution = duration_distribution
        self.reward_type = reward_type
        self.scheduling_policy = scheduling_policy 
        self.metrics = {'application_completion_count': 0,
                        'total_wait_time': 0,
                        'total_server_utilization': 0,
                        'time_steps': 0}
        self.initialize_servers(server_configs) 
        if scheduling_policy == 'rl_agent':
            self.rl_agent = QLearningAgent(self.servers.keys()) 
        else:
            self.rl_agent = None 


    def add_server(self, server):
        self.servers[server.server_id] = server

    def get_server(self, server_id):
        return self.servers.get(server_id)

    def submit_application(self, application):

        self.applications.append(application)
        self.pending_applications.append(application)

    def run_application_on_server(self, application, server):
        if application.run(server, current_time=self.current_time):
            self.pending_applications.remove(application)
            self.running_applications.append(application)
            return True
        return False

    def update_environment(self):
    
        self.generate_new_applications()
        reward = 0 
        apps_to_schedule = [app for app in self.pending_applications if app.arrival_time <= self.current_time] 

        if self.scheduling_policy == 'first_fit':
            reward = self.schedule_applications_first_fit(apps_to_schedule) 
        elif self.scheduling_policy == 'rl_agent':
            reward = self.schedule_applications_rl_agent(apps_to_schedule) 
        else:
            raise ValueError(f"Unknown scheduling policy: {self.scheduling_policy}")


        completed_in_step = []
        for app in list(self.running_applications):
            app.update_status(self.current_time)
            if app.status == "Completed":
                completed_in_step.append(app)

        for app in completed_in_step:
            self.running_applications.remove(app)
            self.completed_applications.append(app)
            app.complete(self)
            self.metrics['application_completion_count'] += 1

        self.current_time += 1
        self.metrics['time_steps'] += 1
        return reward 


    def initialize_servers(self, server_configurations):
    
        for config in server_configurations:
            server = Server(server_id=config['id'], cpu_capacity=config['cpu_capacity'], memory_capacity=config['memory_capacity'])
            self.add_server(server)

    def get_server_utilization_summary(self):
    
        utilization_summary = {}
        for server_id, server in self.servers.items():
            utilization_summary[server_id] = server.get_current_utilization()
        return utilization_summary

    def generate_application_requests(self, distribution_type, params):
    
        if distribution_type == 'fixed':
            return params[0]
        elif distribution_type == 'uniform':
            return random.uniform(params[0], params[1])
        elif distribution_type == 'exponential':
            return np.random.exponential(scale=params[0])
        elif distribution_type == 'lognormal':
            return np.random.lognormal(mean=params[0], sigma=params[1])
        else:
            raise ValueError(f"Unknown distribution type: {distribution_type}")

    def generate_new_applications(self):
        if random.random() < self.arrival_rate:
            self.application_counter += 1
            app_id = f"App-{self.application_counter}"
            cpu_request = int(max(1, self.generate_application_requests(self.cpu_request_distribution[0], self.cpu_request_distribution[1])))
            memory_request = int(max(1, self.generate_application_requests(self.memory_request_distribution[0], self.memory_request_distribution[1])))
            duration = int(max(1, self.generate_application_requests(self.duration_distribution[0], self.duration_distribution[1])))
            arrival_time = self.current_time
            new_application = Application(app_id, cpu_request, memory_request, duration, arrival_time)
            self.submit_application(new_application)
            print(f"Generated and submitted {new_application} at time {self.current_time}") 

    def calculate_reward(self):
        reward = 0
        if self.reward_type == 'utilization_balancing':
            server_utilizations = [sum(server.get_current_utilization()) for server in self.servers.values()]
            utilization_std = np.std(server_utilizations) if server_utilizations else 0
            reward = -utilization_std

        elif self.reward_type == 'latency_focused':
            avg_wait_time = np.mean([app.wait_time for app in self.applications if app.status == 'Pending']) if any(app.status == 'Pending' for app in self.applications) else 0
            reward = -avg_wait_time

        elif self.reward_type == 'resource_efficiency':
            total_utilization = sum(sum(server.get_current_utilization()) for server in self.servers.values())
            num_pending_apps = len(self.pending_applications)
            reward = total_utilization - num_pending_apps

        elif self.reward_type == 'custom':
            reward = 0 

        else:
            raise ValueError(f"Unknown reward type: {self.reward_type}")
        return reward

    def calculate_metrics(self):
        self.metrics['total_wait_time'] = sum(app.wait_time for app in self.completed_applications)
        self.metrics['average_wait_time'] = (self.metrics['total_wait_time'] / self.metrics['application_completion_count']) if self.metrics['application_completion_count'] > 0 else 0
        current_total_utilization = sum(sum(server.get_current_utilization()) for server in self.servers.values())
        self.metrics['total_server_utilization'] += current_total_utilization

    def get_average_utilization(self):
        if self.metrics['time_steps'] > 0:
            return self.metrics['total_server_utilization'] / (self.metrics['time_steps'] * len(self.servers))
        return 0

    def get_metrics_summary(self):
        self.calculate_metrics()
        summary = {
            'application_completion_count': self.metrics['application_completion_count'],
            'average_wait_time': self.metrics['average_wait_time'],
            'average_server_utilization': self.get_average_utilization(),
        }
        return summary

    def get_state(self, application):
        state = [application.cpu_request, application.memory_request]
        
        sorted_server_ids = sorted(self.servers.keys()) 
        for server_id in sorted_server_ids:
            server = self.servers[server_id]
            cpu_util, mem_util = server.get_current_utilization()
            state.extend([cpu_util, mem_util]) 
        return tuple(state) 

    def schedule_applications_first_fit(self, applications_to_schedule):
        reward = 0 
        for app in applications_to_schedule:
            if app.status == "Pending":
                print(f"Attempting to schedule {app.application_id} (CPU: {app.cpu_request}, Memory: {app.memory_request}, Duration: {app.duration})...")
                for server_id, server in self.servers.items():
                    if self.run_application_on_server(app, server):
                        print(f"{app.application_id} started on {server_id}.")
                        break
                else:
                    print(f"Could not schedule {app.application_id} - no suitable server found.")
        return self.calculate_reward() 


    def schedule_applications_rl_agent(self, applications_to_schedule):
    
        total_reward = 0 
        for app in applications_to_schedule:
            if app.status == "Pending":
                print(f"Attempting to schedule {app.application_id} (CPU: {app.cpu_request}, Memory: {app.memory_request}, Duration: {app.duration}) using RL Agent...")
                state = self.get_state(app) 
                action = self.rl_agent.choose_action(state, self.get_valid_actions_for_app(app)) 
                if action is not None: 
                    server_id_to_assign = list(self.servers.keys())[action] 
                    server_to_assign = self.servers[server_id_to_assign]
                    if self.run_application_on_server(app, server_to_assign): 
                        print(f"{app.application_id} started on {server_id_to_assign} based on RL agent decision.")
                        
                    else:
                        print(f"RL Agent chose Server {server_id_to_assign} for {app.application_id}, but allocation failed (insufficient resources - shouldn't happen if valid actions are used).")
                else:
                    print(f"RL Agent chose to not assign {app.application_id} (action None). Keeping it pending.") 

        return self.calculate_reward() 


    def get_valid_actions_for_app(self, application):
        valid_actions = []
        server_ids_sorted = sorted(self.servers.keys()) 
        for action_index, server_id in enumerate(server_ids_sorted): 
            server = self.servers[server_id]
            if server.get_available_cpu() >= application.cpu_request and server.get_available_memory() >= application.memory_request:
                valid_actions.append(action_index) 
        return valid_actions


    def __str__(self):
        return (f"CloudEnv: {len(self.servers)} servers, Total Apps: {len(self.applications)} "
                f"(Pending: {len(self.pending_applications)}, Running: {len(self.running_applications)}, Completed: {len(self.completed_applications)}), Time: {self.current_time}, "
                f"Reward Type: {self.reward_type}, Policy: {self.scheduling_policy}")


class QLearningAgent:
    def __init__(self, server_ids, learning_rate=0.1, discount_factor=0.9, exploration_rate=0.1):
        self.server_ids = sorted(list(server_ids)) 
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.q_table = {} 

    def choose_action(self, state, valid_actions_indices):
        if not valid_actions_indices: 
            return None 

        if random.uniform(0, 1) < self.exploration_rate: 
            return random.choice(valid_actions_indices) 
        else: 
            if state not in self.q_table: 
                self.q_table[state] = {action_index: 0 for action_index in valid_actions_indices}
                return random.choice(valid_actions_indices) 
            else:
                q_values = self.q_table[state]
                best_action = max(valid_actions_indices, key=lambda action_index: q_values.get(action_index, 0)) 
                return best_action

    def learn(self, state, action_index, reward, next_state, next_valid_actions_indices):
        if state not in self.q_table:
            self.q_table[state] = {action_index_init: 0 for action_index_init in range(len(self.server_ids))} 

        if next_state is None: 
            max_next_q_value = 0 
        else:
            if next_state not in self.q_table:
                 self.q_table[next_state] = {action_index_init: 0 for action_index_init in range(len(self.server_ids))} 
            if next_valid_actions_indices: 
                max_next_q_value = max(self.q_table[next_state].get(action_index_next, 0) for action_index_next in next_valid_actions_indices) 
            else:
                max_next_q_value = 0 


        current_q_value = self.q_table[state].get(action_index, 0) 
        new_q_value = current_q_value + self.learning_rate * (reward + self.discount_factor * max_next_q_value - current_q_value) 
        self.q_table[state][action_index] = new_q_value 



if __name__ == "__main__":
    
    server_configs = [
        {'id': 'Server-1', 'cpu_capacity': 100, 'memory_capacity': 200},
        {'id': 'Server-2', 'cpu_capacity': 80, 'memory_capacity': 150}
    ]
    arrival_rate = 0.7
    cpu_request_distribution = ('uniform', [10, 40])
    memory_request_distribution = ('uniform', [20, 80])
    duration_distribution = ('fixed', [5])
    simulation_duration = 200
    reward_type = 'resource_efficiency'
    scheduling_policy = 'rl_agent' 


    
    environment = CloudEnvironment(server_configs=server_configs,
                                     arrival_rate=arrival_rate,
                                     cpu_request_distribution=cpu_request_distribution,
                                     memory_request_distribution=memory_request_distribution,
                                     duration_distribution=duration_distribution,
                                     reward_type=reward_type,
                                     scheduling_policy=scheduling_policy) 
    print(f"Cloud Environment initialized with servers: {list(environment.servers.keys())}, Reward Type: {environment.reward_type}, Scheduling Policy: {environment.scheduling_policy}")

    
    total_reward = 0
    for _ in range(simulation_duration):
        print(f"\n--- Time Step: {environment.current_time} ---")
        print(f"Environment Summary: {environment}")
        server_utilizations = environment.get_server_utilization_summary()
        for server_id, utilization in server_utilizations.items():
            print(f"  {server_id} Utilization - CPU: {utilization[0]:.2f}%, Memory: {utilization[1]:.2f}%")

        
        apps_to_schedule = [app for app in environment.pending_applications if app.arrival_time <= environment.current_time] 

        if environment.scheduling_policy == 'rl_agent': 
            for app in list(apps_to_schedule): 
                if app.status == "Pending":
                    state = environment.get_state(app) 
                    valid_actions = environment.get_valid_actions_for_app(app) 
                    action_index = environment.rl_agent.choose_action(state, valid_actions) 

                    server_id_chosen = None 
                    if action_index is not None: 
                        server_id_chosen = list(environment.servers.keys())[action_index] 
                        server_to_assign = environment.servers[server_id_chosen]
                        environment.run_application_on_server(app, server_to_assign) 

                    reward = environment.update_environment() 
                    total_reward += reward 

                    next_state = environment.get_state(app) 
                    next_valid_actions = environment.get_valid_actions_for_app(app) 

                    if action_index is not None: 
                         environment.rl_agent.learn(state, action_index, reward, next_state, next_valid_actions) 

                    print(f"RL Agent scheduled {app.application_id}. Action: Server '{server_id_chosen if server_id_chosen else 'None (not assigned)'}', Reward: {reward:.2f}, Total Reward: {total_reward:.2f}") 

        elif environment.scheduling_policy == 'first_fit': 
            reward = environment.update_environment() 
            total_reward += reward 
            environment.schedule_applications_first_fit(apps_to_schedule) 
            print(f"First-Fit policy scheduled apps. Reward: {reward:.2f}, Total Reward: {total_reward:.2f}") 

        else: 
            reward = environment.update_environment() 
            total_reward += reward 
            print(f"No Scheduling Policy Active. Reward: {reward:.2f}, Total Reward: {total_reward:.2f}") 


    
    print("\n--- Simulation End ---")
    print(f"Final Environment Summary: {environment}")
    metrics_summary = environment.get_metrics_summary()
    print("\nMetrics Summary:")
    for key, value in metrics_summary.items():
        print(f"  {key}: {value:.2f}")
    print("\nCompleted Applications Summary ({len(environment.completed_applications)}):")
    for app in environment.completed_applications[-10:]: 
        print(app)
    print(f"... (Total {len(environment.completed_applications)} completed applications)")
    print("\nRunning Applications Summary ({len(environment.running_applications)}):")
    for app in environment.running_applications:
        print(app)
    print("\nPending Applications Summary ({len(environment.pending_applications)}):")
    for app in environment.pending_applications:
        print(app)
    final_server_utilizations = environment.get_server_utilization_summary()
    print("\nFinal Server Utilizations:")
    for server_id, utilization in final_server_utilizations.items():
        print(f"  {server_id} Utilization - CPU: {utilization[0]:.2f}%, Memory: {utilization[1]:.2f}%")
    print("\n--- Q-Table (First 10 entries): ---") 
    q_table_items = list(environment.rl_agent.q_table.items())
    for state, action_values in q_table_items[:10]:
        print(f"State: {state}, Q-Values: {action_values}")
    if not q_table_items:
        print("Q-table is empty (no learning happened yet or very short simulation).")