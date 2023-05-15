import numba

class Sampler:
    def __init__(self, process, repeats):
        self.repeats = repeats
        self.processes = [process.copy(f'{process.name}_{i}') for i in range(self.repeats)]
    
    def run(self, nb_steps):
        for i in numba.prange(self.repeats):
            for _ in range(nb_steps):
                self.processes[i].update()
    
