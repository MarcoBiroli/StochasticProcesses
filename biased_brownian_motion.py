from stochastic_process import StochasticProcess
from sampling import Sampler
import numpy as np
import argparse
import numba

def parse():
    parser = argparse.ArgumentParser(description='Simulate critical walkers.')
    parser.add_argument('--variable_number', type = int, default = 1)
    parser.add_argument('--variable_dimension', type = int, default = 1)
    parser.add_argument('--drift', type = float, default = 1)
    parser.add_argument('--diffusion', type = float, default = 1)
    parser.add_argument('--time_step', type = float, default = 0.1)
    parser.add_argument('--record_trajectory', action='store_true')
    return parser.parse_args()

def create_update_function(variable_number, variable_dimension, drift, diffusion, time_step):
    def wrapper(variables, time = None):
        return biased_random_walk(variable_number = variable_number,
                                  variable_dimension = variable_dimension,
                                  drift = drift,
                                  diffusion = diffusion,
                                  time_step = time_step,
                                  variables = variables)
    return wrapper


def biased_random_walk(variable_number, variable_dimension, drift, diffusion, time_step, variables):
    assert variables.ndim == 2, 'Variables has the wrong number of dimensions.'
    assert variables.shape == (variable_number, variable_dimension), 'Variables and variable shapes do not match.'

    noise = np.sqrt(2 * diffusion * time_step) * np.random.normal(size = (variable_number, variable_dimension))

    return variables + noise + drift * time_step

def positive_counter(variables):
    return np.sum((variables > 0).all(axis = 1))

args = parse()

update_function = create_update_function(args.variable_number, 
                                         args.variable_dimension, 
                                         args.drift, 
                                         args.diffusion, 
                                         args.time_step)

process = StochasticProcess(update_function=update_function,
                            time_step=args.time_step,
                            record_trajectory=args.record_trajectory,
                            variable_number = args.variable_number,
                            variable_dimension = args.variable_dimension,
                            observable_flux = positive_counter)


print(process)

sampler = Sampler(process=process, repeats = 100)

sampler.run(100000)
