import datetime
import copy
import pandas as pd
import random

from chromosome import *


"""Create population function - PMSBX-GA"""


def init_population(size_of_population):
    population = []
    for i in range(size_of_population):
        individual = CHROMOSOME_PMSBX_GA(get_data())
        population.append(individual)
    population = np.asarray(population)
    return population


"""Select a sub-population for offspring production - PMSBX-GA"""


def select_mating_pool(pop, num_parents_mating):
    # shuffling the pop then select top of pops
    pop = np.asarray(pop)
    index = np.random.choice(pop.shape[0], num_parents_mating, replace=False)
    random_individual = pop[index]
    # split current pop into remain_pop and mating_pool
    # pop = np.delete(pop, index)
    return random_individual

"""Crossover - PMSBX-GA"""


def crossover(parents):
    pass


"""Mutation - PMSBX-GA"""


def mutation(parents, random_rate):
    pass


"""Selecting a new population for the next generation from parents and offsprings - PMSBX-GA"""


def selection(parents, offsprings):
    pass
