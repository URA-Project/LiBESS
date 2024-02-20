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
    # shuffling the pop then select top of pops
    pop = np.asarray(pop)
    index = np.random.choice(pop.shape[0], num_parents_mating, replace=False)
    random_individual = pop[index]
    # split current pop into remain_pop and mating_pool
    # pop = np.delete(pop, index)
    return random_individual


"""Crossover - GA"""


def crossover(parents):
    # Initializes a numpy array offspring by copying the first half of the parents array. This will be used to store the child solutions
    offspring = np.copy(parents[: parents.shape[0] // 2])
    # The point at which crossover takes place between two parents, which in this case is at the center.
    crossover_point = parents.shape[1] // 2
    # Perform crossover
    for k in range(0, parents.shape[0], 2):
        # Index of the first parent to mate.
        parent1_idx = k % parents.shape[0]
        # Index of the second parent to mate.
        parent2_idx = (k + 1) % parents.shape[0]
        if k < len(offspring):
            # The new offspring will have its first half of its genes taken from the first parent.
            offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
            # The new offspring will have its second half of its genes taken from the second parent.
            offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
    return offspring


"""Mutation - GA"""


# Take every individual in the offspring after crossover to mutate with a given rate
def mutation(offspring_crossover, random_rate):
    # 0 and 1 are genes that can be used to replace the existing genes in the offspring solutions during the mutation operation.
    geneSet = ["0", "1"]

    for offspring in offspring_crossover:
        for task in offspring:
            # Random a rate and check if it's < given rate, then mutate it
            rate = random.uniform(0, 1)
            if rate < random_rate:
                index = random.randrange(task.rfind("-") + 1, len(task))
                newGene, alternate = random.sample(geneSet, 2)
                mutate_gene = alternate if newGene == task[index] else newGene
                task = task[:index] + mutate_gene + task[index + 1 :]

    return offspring_crossover


"""Selecting a new population for the next generation from parents and offsprings - GA"""


def selection(parents, offsprings):  # num individual = num parents
    # Combine parents and offsprings
    population = np.concatenate((parents, offsprings), axis=0)

    # Calculate fitness for each individual in the population
    fitness = np.array(
        [individual.fitness for individual in population]
    )  # individual.fitness = fitness(individual)

    # Select the best individuals for the next generation
    num_parents = parents.shape[0]  # parents.shape[0] = num_parents_mating
    new_population = population[
        fitness.argsort()[-num_parents:]
    ]  # first n-largest fitness

    return new_population
