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
    pass


"""Crossover - PMSBX-GA"""


def crossover(parents):
    pass


"""Mutation - PMSBX-GA"""


def mutation(parents, random_rate):
    pass


"""Selecting a new population for the next generation from parents and offsprings - PMSBX-GA"""


def selection(parents, offsprings):
    pass
