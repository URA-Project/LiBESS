import ga
import random
import numpy as np
import pandas as pd
import statistics
import sys


class GA_Algorithm:
    def __init__(self, popsize, num_parents_mating, num_generations, mutation_rate):
        # Need to update
        self.popsize = popsize
        self.num_parents_mating = num_parents_mating
        self.num_generations = num_generations
        self.mutation_rate = mutation_rate
        self.population = ga.createParent(self.sol_per_pop)
        self.pop_size = self.population.shape
        self.best_outputs = []

    def run_algorithm(self):
        # Add the steps of running the GA algorithm.
        pass


class PMSBX_GA_Algorithm:
    def __init__(self, popsize, num_parents_mating, num_generations, mutation_rate):
        # Need to update
        self.popsize = popsize
        self.num_parents_mating = num_parents_mating
        self.num_generations = num_generations
        self.mutation_rate = mutation_rate
        self.population = ga.createParent(self.sol_per_pop)
        self.pop_size = self.population.shape
        self.best_outputs = []

    def run_algorithm(self):
        # Add the steps of running the PMSBX-GA algorithm.
        pass
