

import random
from task import Task

def generate_tasks_uniform(time_step, num_tasks=5, max_cpu=4, max_memory=16, max_duration=10, max_gpu=2):

    tasks = []
    for i in range(num_tasks):
        cpu_req = random.randint(1, max_cpu)
        mem_req = random.randint(1, max_memory)
        duration = random.randint(1, max_duration)
        gpu_req = random.randint(0, max_gpu)

        task_id = f"{time_step}_{i}"
        task = Task(
            task_id=task_id,
            cpu_requirement=cpu_req,
            memory_requirement=mem_req,
            duration=duration,
            gpu_requirement=gpu_req
        )
        tasks.append(task)
    return tasks

def generate_tasks_bursty(time_step, burst_probability=0.1, max_burst_size=20, max_cpu=4, max_memory=16, max_duration=10, max_gpu=2):

    tasks = []
    if random.uniform(0, 1) < burst_probability:
        num_tasks = random.randint(5, max_burst_size)
    else:
        num_tasks = random.randint(0, 2)

    for i in range(num_tasks):
        cpu_req = random.randint(1, max_cpu)
        mem_req = random.randint(1, max_memory)
        duration = random.randint(1, max_duration)
        gpu_req = random.randint(0, max_gpu)

        task_id = f"{time_step}_{i}"
        task = Task(
            task_id=task_id,
            cpu_requirement=cpu_req,
            memory_requirement=mem_req,
            duration=duration,
            gpu_requirement=gpu_req
        )
        tasks.append(task)
    return tasks

def generate_tasks_with_pattern(time_step):

    tasks = []
    
    if 20 <= time_step <= 30:
        num_tasks = 10
    else:
        num_tasks = 2

    for i in range(num_tasks):
        cpu_req = random.randint(1, 8)
        mem_req = random.randint(1, 32)
        duration = random.randint(1, 15)
        gpu_req = random.randint(0, 4)

        task_id = f"{time_step}_{i}"
        task = Task(
            task_id=task_id,
            cpu_requirement=cpu_req,
            memory_requirement=mem_req,
            duration=duration,
            gpu_requirement=gpu_req
        )
        tasks.append(task)
    return tasks


