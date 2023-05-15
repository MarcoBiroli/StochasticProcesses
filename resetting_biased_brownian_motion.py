from stochastic_process import EarlyStoppingStochasticProcess
from sampling import Sampler
import numpy as np
import argparse
import numba
import matplotlib.pyplot as plt

def parse():
    parser = argparse.ArgumentParser(description='Simulate critical walkers.')
    parser.add_argument('--variable_number', type = int, default = 1)
    parser.add_argument('--variable_dimension', type = int, default = 1)
    parser.add_argument('--drift', type = float, default = 1)
    parser.add_argument('--diffusion', type = float, default = 1)
    parser.add_argument('--time_step', type = float, default = 0.1)
    parser.add_argument('--record_trajectory', action='store_true')
    parser.add_argument('--reset_rate', type=float, default=1)
    parser.add_argument('--reset_position', type=float, default=0)
    return parser.parse_args()

def create_update_function(variable_number, variable_dimension, drift, diffusion, reset_rate, reset_position, time_step):
    def wrapper(variables, time = None):
        return resetting_biased_random_walk(variable_number = variable_number,
                                  variable_dimension = variable_dimension,
                                  drift = drift,
                                  diffusion = diffusion,
                                  reset_rate = reset_rate,
                                  reset_position = reset_position,
                                  time_step = time_step,
                                  variables = variables)
    return wrapper


def resetting_biased_random_walk(variable_number, variable_dimension, drift, diffusion, reset_rate, reset_position, time_step, variables):
    assert variables.ndim == 2, 'Variables has the wrong number of dimensions.'
    assert variables.shape == (variable_number, variable_dimension), 'Variables and variable shapes do not match.'

    if np.random.rand() < reset_rate * time_step:
        return variables*0 + reset_position

    noise = np.sqrt(2 * diffusion * time_step) * np.random.normal(size = (variable_number, variable_dimension))

    return variables + noise + drift * time_step

def positive_flag(variables):
    return (variables > 0).any()


args = parse()

update_function = create_update_function(args.variable_number, 
                                         args.variable_dimension, 
                                         args.drift, 
                                         args.diffusion, 
                                         args.reset_rate,
                                         args.reset_position,
                                         args.time_step)

process = EarlyStoppingStochasticProcess(stopping_criterion = positive_flag,
                                         update_function=update_function,
                                         time_step=args.time_step,
                                         record_trajectory=args.record_trajectory,
                                         variable_number = args.variable_number,
                                         variable_dimension = args.variable_dimension,
                                         initial_variable = -5
                                         )


print(process)

sampler = Sampler(process=process, repeats = 1)

sampler.run(10000)

averages= sampler.get_averages()
print(averages.keys())
print(averages)

