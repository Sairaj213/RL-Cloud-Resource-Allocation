

class Server:
    def __init__(self, server_id, cpu_cores, memory_gb, gpu_cores=0):

        self.server_id = server_id
        self.total_cpu_cores = cpu_cores
        self.total_memory_gb = memory_gb
        self.total_gpu_cores = gpu_cores

        self.available_cpu_cores = cpu_cores
        self.available_memory_gb = memory_gb
        self.available_gpu_cores = gpu_cores

        self.tasks = []

    def allocate_resources(self, task):

        if (self.available_cpu_cores >= task.cpu_requirement and
            self.available_memory_gb >= task.memory_requirement and
            self.available_gpu_cores >= task.gpu_requirement):

            self.available_cpu_cores -= task.cpu_requirement
            self.available_memory_gb -= task.memory_requirement
            self.available_gpu_cores -= task.gpu_requirement

            self.tasks.append(task)
            task.is_allocated = True
            task.server_id = self.server_id

            return True
        else:
            return False

    def release_resources(self, task):
        if task in self.tasks:
            self.available_cpu_cores += task.cpu_requirement
            self.available_memory_gb += task.memory_requirement
            self.available_gpu_cores += task.gpu_requirement

            self.tasks.remove(task)
            task.is_allocated = False
            task.server_id = None

    def get_utilization(self):
        cpu_util = (self.total_cpu_cores - self.available_cpu_cores) / self.total_cpu_cores * 100
        mem_util = (self.total_memory_gb - self.available_memory_gb) / self.total_memory_gb * 100
        gpu_util = 0
        if self.total_gpu_cores > 0:
            gpu_util = (self.total_gpu_cores - self.available_gpu_cores) / self.total_gpu_cores * 100

        return {
            'cpu': cpu_util,
            'memory': mem_util,
            'gpu': gpu_util
        }

    def __str__(self):
        return f"Server {self.server_id}: CPU {self.available_cpu_cores}/{self.total_cpu_cores} cores, " \
               f"Memory {self.available_memory_gb}/{self.total_memory_gb} GB, " \
               f"GPU {self.available_gpu_cores}/{self.total_gpu_cores} cores"

    def __repr__(self):
        return self.__str__()
