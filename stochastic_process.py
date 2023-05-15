from copy import deepcopy
import numpy as np
import re

class StochasticProcess:
    '''
    ...
    '''
    def __init__(self, 
                 update_function, 
                 time_step, 
                 record_trajectory = False, 
                 known_time_steps = None,
                 buffer_size = 100,
                 **kwargs):

        if (not('variable_number' in kwargs and 'variable_dimension' in kwargs) and (not 'initial_variables' in kwargs)):
            raise ValueError('The process cannot be initialized without specifying either \
                             the number of variables and their dimension or the initial variables.')
        
        if 'variable_number' in kwargs and 'variable_dimension' in kwargs:
            self.variable_number = kwargs['variable_number']
            self.variable_dimension = kwargs['variable_dimension']
            self.variables = np.zeros(shape=(self.variable_number, self.variable_dimension))

        if 'initial_variables' in kwargs:
            self.variables = np.array(kwargs['initial_variables'])
            if self.variables.ndim == 1: self.variables = self.variables[..., np.newaxis]
            assert self.variables.ndim == 2, 'Incorrect number of dimensions for the initial variables.'

            self.variable_number, self.variable_dimension = self.variables.shape

        if 'name' in kwargs:
            self.name = kwargs['name']
        else:
            self.name = 'Stochastic process'
        
        self.buffer_size = buffer_size
        self.known_time_steps = known_time_steps
        if not(known_time_steps is None): self.buffer_size = self.known_time_steps + 1

        self.observables = {}
        self.results = {}
        self.result_buffers = {}
        self.result_buffer_positions = {}
        for key, value in kwargs.items():
            match = re.match(r'(\w+)\_(\w+)', key)
            if not match: continue
            if match.group(1) != 'observable': continue

            observable_name = match.group(2)
            self.observables[observable_name] = value
            first_observation = value(self.variables)
            self.results[observable_name] = np.empty(shape = (0,) + first_observation.shape, dtype = first_observation.dtype)
            self.result_buffers[observable_name] = np.empty(shape = (self.buffer_size,) + first_observation.shape, dtype = first_observation.dtype)
            self.result_buffer_positions[observable_name] = 0

        self.observable_names = list(self.observables.keys())

        self.update_function = update_function
        self.time = 0
        self.time_step = time_step

        self.record_trajectory = record_trajectory

        if self.record_trajectory:
            self.buffer = np.empty(shape=(self.buffer_size, self.variable_number, self.variable_dimension), dtype = self.variables.dtype)
            self.buffer[0] = deepcopy(self.variables)
            self.buffer_position = 1
        
        self.trajectory = np.empty(shape=(0, self.variable_number, self.variable_dimension), dtype = self.variables.dtype)

        self.measure()

    def __str__(self):
        return (f'{self.name}\n'
              f'---------------------------- \n'
              f'Number of variables: {self.variable_number} \n'
              f'Dimension of variables: {self.variable_dimension} \n'
              f'Time step: {self.time_step} \n'
              f'Current time: {self.time} \n'
              f'Current variables: {self.variables} \n'
              f'Observables: {list(self.observables.keys())}')

    def __print__(self):
        print(str(self))

    def copy(self, new_name = None):
        if new_name is None:
            new_name = self.name + '_copy'
        return StochasticProcess(update_function=self.update_function,
                                 time_step=self.time_step,
                                 record_trajectory=self.record_trajectory,
                                 known_time_steps=self.known_time_steps,
                                 buffer_size=self.buffer_size,
                                 variable_number = self.variable_number,
                                 variable_dimension = self.variable_dimension,
                                 initial_variables = deepcopy(self.variables),
                                 name=new_name)

    def update(self):
        self.variables = self.update_function(
            variables = self.variables,
            time = self.time)
        self.time = self.time + self.time_step

        self.measure()

        if not self.record_trajectory:
            return
        
        if self.buffer_position == self.buffer_size: 
            self.trajectory = np.concatenate((self.trajectory, self.buffer), axis = 0)
            self.buffer_position = 0
        else:
            self.buffer[self.buffer_position] = self.variables
            self.buffer_position += 1

    
    def measure(self):
        for observable_name, observable in self.observables.items():
            if self.result_buffer_positions[observable_name] == self.buffer_size:
                self.results[observable_name] = np.concatenate((self.results[observable_name], self.result_buffers[observable_name]), axis = 0)
                self.result_buffer_positions[observable_name] = 0
            else:
                self.result_buffers[observable_name][self.result_buffer_positions[observable_name]] = observable(self.variables)
                self.result_buffer_positions[observable_name] += 1

    def get_result(self, observable_name, times = None):
        result = self.results[observable_name]
        if times is None:
            return result
        
        if type(times) == int:
            assert times in result, f'Requested time, t = {times} has not been computed for {observable_name}.'
            return result[times]
        
        out = {}
        for time in times:
            assert time in result, f'Requested time, t = {time} has not been computed for {observable_name}.'
            out[time] = result[time]

        return out
    
    def get_trajectory(self):
        assert self.record_trajectory, 'Cannot retrieve trajectory if it has not been computed.'
        if self.buffer_position != 0:
            self.buffer = self.buffer[:self.buffer_position]
        self.trajectory = np.concatenate((self.trajectory, self.buffer), axis = 0)
        return {self.time_step * index : value for index, value in enumerate(self.trajectory)}

