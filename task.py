

class Task:
    def __init__(self, task_id, cpu_requirement, memory_requirement, duration, gpu_requirement=0):
    
        self.task_id = task_id
        self.cpu_requirement = cpu_requirement
        self.memory_requirement = memory_requirement
        self.gpu_requirement = gpu_requirement
        self.duration = duration  

        self.is_allocated = False
        self.server_id = None

    def step(self, time_unit=1):
    
        if self.is_allocated:
            self.duration -= time_unit
            if self.duration <= 0:
                self.duration = 0
                return True  
            else:
                return False  
        else:
            return False  

    def __str__(self):
        return f"Task {self.task_id}: CPU {self.cpu_requirement} cores, " \
               f"Memory {self.memory_requirement} GB, " \
               f"GPU {self.gpu_requirement} cores, " \
               f"Duration {self.duration} units, " \
               f"Allocated: {self.is_allocated}, Server: {self.server_id}"

    def __repr__(self):
        return self.__str__()
