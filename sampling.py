import numba
from numba_progress import ProgressBar

class Sampler:
    def __init__(self, process, repeats):
        self.repeats = repeats
        self.processes = [process.copy(f'{process.name}_{i}') for i in range(self.repeats)]
    
    def run(self, nb_steps):
        with ProgressBar(total = self.repeats*nb_steps, leave = False) as progress:
            for i in numba.prange(self.repeats):
                for _ in range(nb_steps):
                    self.processes[i].update()
                    progress.update(1)
    
    def get_averages(self):
        averages = {}
        for observable_name in self.processes[0].observable_names:
            averages[observable_name] = self.processes[0].get_result(observable_name)
            for i in range(1, self.repeats):
                averages[observable_name] += self.processes[i].get_result(observable_name)
        return averages
