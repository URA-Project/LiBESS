import datetime
import copy
import pandas as pd
import random

from chromosome import *
from utils import *

"""Create population function - GA"""


def init_population(size_of_population):
    population = []
    for i in range(size_of_population):
        individual = CHROMOSOME_GA(get_data())
        population.append(individual)
    population = np.asarray(population)
    return population


"""Select a sub-population for offspring production - GA"""


def select_mating_pool(pop, num_parents_mating):
    pass


"""Crossover - GA"""


def crossover(parents):
    pass


"""Mutation - GA"""


def mutation(parents, random_rate):
    pass


"""Selecting a new population for the next generation from parents and offsprings - GA"""


def selection(parents, offsprings):
    pass
