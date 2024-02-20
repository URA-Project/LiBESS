import ga
import pmsbx_ga
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
        population = ga.init_population(self.popsize)
        # Add code evalue fitnesss
        for generation in range(self.num_generations):
            # Selecting the best parents in the population for mating.
            parents = ga.select_mating_pool(population, self.num_parents_mating)
            # Crossover
            offspring_crossover = ga.crossover(parents)
            # Take every individual in the offspring after crossover to mutate with a given rate
            offspring_mutation = ga.mutation(offspring_crossover)
            # Selecting a new population for the next generation from parents and offsprings
            population = ga.selection(parents, offspring_mutation)


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
        population = pmsbx_ga.init_population(self.popsize)
        # Add code evalue fitnesss
        for generation in range(self.num_generations):
            # Selecting the best parents in the population for mating.
            parents = pmsbx_ga.select_mating_pool(population, self.num_parents_mating)
            # Crossover
            offspring_crossover = pmsbx_ga.crossover(parents)
            # Take every individual in the offspring after crossover to mutate with a given rate
            offspring_mutation = pmsbx_ga.mutation(offspring_crossover)
            # Selecting a new population for the next generation from parents and offsprings
            population = pmsbx_ga.selection(parents, offspring_mutation)
