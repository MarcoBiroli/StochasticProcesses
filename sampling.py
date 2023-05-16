import numba
from numba_progress import ProgressBar
import numpy as np

class Sampler:
    """
    The Sampler class is a simple wrapper allowing to repeat the execution
    of a StochasticProcess multiple times in order to compute averages of 
    the observables computed by the StochasticProcess.
    """
    def __init__(self, process, repeats : int):
        """
        Initializes a instance of the Sampler class.

        Args:
            process (StochasticProcess): The process to sample from.
            repeats (int): The number of copies of the process to be made.
        """
        self.repeats = repeats
        self.processes = [process.copy(f'{process.name}_{i}') for i in range(self.repeats)]
    
    def run(self, nb_steps : int):
        """
        The run method updates each process for the given number of steps.

        Args:
            nb_steps (int): The number of steps to perform for each process.
        """
        with ProgressBar(total = self.repeats*nb_steps, leave = False) as progress:
            for i in numba.prange(self.repeats):
                for _ in range(nb_steps):
                    self.processes[i].update()
                    progress.update(1)
    
    def get_averages(self) -> dict[str, np.ndarray]:
        """
        Compute the average of the observables of the process.

        Returns:
            dict[str, np.ndarray]: A dictionnary containing the computed
            averages.
        """
        averages = {}
        for observable_name in self.processes[0].observable_names:
            averages[observable_name] = self.processes[0].get_result(observable_name)
            for i in range(1, self.repeats):
                averages[observable_name] += self.processes[i].get_result(observable_name)
        return averages

class EarlyStoppingSampler(Sampler):
    """
    The EarlyStoppingSampler class is a simple wrapper allowing to repeat the execution
    of a EarlyStoppingStochasticProcess multiple times in order to compute averages of 
    the observables computed by the EarlyStoppingStochasticProcess.
    """
    def __init__(self, process, repeats):
        """
        Initializes a instance of the EarlyStoppingSampler class.

        Args:
            process (EarlyStoppingStochasticProcess): The process to sample from.
            repeats (int): The number of copies of the process to be made.
        """
        super().__init__(process, repeats)
    
    def run(self, nb_steps : int) -> dict[str, np.ndarray]:
        """
        The run method updates each process for the given number of steps, unless
        the process gets stopped early by it's stopping condition.

        Args:
            nb_steps (int): The number of steps to perform for each process.
        """
        with ProgressBar(total = self.repeats*nb_steps, leave = False) as progress:
            for i in numba.prange(self.repeats):
                for cur_step in range(nb_steps):
                    self.processes[i].update()
                    flag = self.processes[i].stopping_condition()
                    if flag:
                        progress.update(nb_steps - cur_step)
                        break
                    progress.update(1)
