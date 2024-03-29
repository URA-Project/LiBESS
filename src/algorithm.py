from src import ga, pmsbx_ga
from src.conditions_pmsbx_ga import cal_fitness_value
import src.ga
import src.pmsbx_ga
import random
import numpy as np
import pandas as pd
import statistics
import sys


class GA_Algorithm:
    def __init__(
        self,
        popsize,
        num_parents_mating,
        num_generations,
        mutation_rate,
        HC_penalt_point,
        SC_penalt_point,
    ):
        # Need to update
        self.popsize = popsize
        self.num_parents_mating = num_parents_mating
        self.num_generations = num_generations
        self.mutation_rate = mutation_rate
        self.best_outputs = []
        self.HC_penalt_point = HC_penalt_point
        self.SC_penalt_point = SC_penalt_point

    def run_algorithm(self):
        population = ga.init_population(self.popsize)
        # Add code evalue fitnesss
        for generation in range(self.num_generations):
            # Selecting the best parents in the population for mating.
            parents = ga.select_mating_pool(population, self.num_parents_mating)
            # Crossover
            offspring_crossover = ga.crossover(parents)
            # Take every individual in the offspring after crossover to mutate with a given rate
            offspring_mutation = ga.mutation(offspring_crossover, random_rate=0.5)
            # Selecting a new population for the next generation from parents and offsprings
            population = ga.selection(
                parents, offspring_mutation, self.HC_penalt_point, self.SC_penalt_point
            )


class PMSBX_GA_Algorithm:
    def __init__(
        self,
        popsize,
        num_parents_mating,
        num_generations,
        distribution_index,
        HC_penalt_point,
        SC_penalt_point,
    ):
        # Need to update
        self.popsize = popsize
        self.num_parents_mating = num_parents_mating
        self.num_generations = num_generations
        self.distribution_index = distribution_index
        self.best_outputs = []
        self.HC_penalt_point = HC_penalt_point
        self.SC_penalt_point = SC_penalt_point

    def run_algorithm(self):
        # Add the steps of running the PMSBX-GA algorithm.
        population = pmsbx_ga.init_population(self.popsize)
        # Add code evalue fitnesss
        for generation in range(self.num_generations):
            # Selecting the best parents in the population for mating.
            parents = pmsbx_ga.select_mating_pool(population, self.num_parents_mating)
            # Crossover
            offspring_crossover = pmsbx_ga.crossover(parents, self.distribution_index)
            # Take every individual in the offspring after crossover to mutate with a given rate
            offspring_mutation = pmsbx_ga.mutation(
                offspring_crossover, self.distribution_index
            )
            # Selecting a new population for the next generation from parents and offsprings
            population = pmsbx_ga.selection(
                parents, offspring_mutation, self.HC_penalt_point, self.SC_penalt_point
            )


def main():
    # Các thông số của thuật toán
    popsize = 6
    num_parents_mating = 4
    num_generations = 3000
    distribution_index = 100
    HC_penalt_point = 10
    SC_penalt_point = 3

    # Tạo một đối tượng PMSBX_GA_Algorithm
    pmsbx_ga_algorithm = PMSBX_GA_Algorithm(
        popsize,
        num_parents_mating,
        num_generations,
        distribution_index,
        HC_penalt_point,
        SC_penalt_point,
    )

    # Chạy thuật toán
    pmsbx_ga_algorithm.run_algorithm()


if __name__ == "__main__":
    main()
