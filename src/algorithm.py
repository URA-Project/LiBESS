import copy

from src.NSGA import nsga
from src.PMSBX_GA import pmsbx_ga
from src.GA import ga
import numpy as np
import pandas as pd
import statistics
import sys
import random

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


class NSGA_ALGORITHM:
    def __init__(self, popsize, num_parents_mating, num_generations, HC_penalt_point, SC_penalt_point):
        # Need to update
        self.popsize = popsize
        self.num_parents_mating = num_parents_mating
        self.num_generations = num_generations
        self.best_outputs = []
        self.HC_penalt_point = HC_penalt_point
        self.SC_penalt_point = SC_penalt_point

    def run_algorithm(self):
        # Add the steps of running the NSGA algorithm.
        best_list, best_obj = [], []
        population = nsga.init_population(self.popsize)
        # Add code evalue fitnesss
        for generation in range(self.num_generations):
            # Selecting the best parents in the population for mating.
            parents = nsga.select_mating_pool(population, self.num_parents_mating)
            # Crossover
            offspring_crossover = nsga.crossover(parents)
            # Take every individual in the offspring after crossover to mutate with a given rate
            offspring_mutation = nsga.mutation(offspring_crossover)
            chroms_obj_record = {}  # record each chromosome objective values as chromosome_obj_record={chromosome:[HC_time,HC_record]}

            total_chromosome = np.concatenate((copy.deepcopy(parents), copy.deepcopy(
                offspring_crossover), copy.deepcopy(population), copy.deepcopy(
                offspring_mutation)))  # combine parent and offspring chromosomes
            for m in range(self.popsize * 2):
                HC_time, HC_resource = nsga.fitness_value(total_chromosome[m])
                chroms_obj_record[m] = [HC_time, HC_resource]
            '''-------non-dominated sorting-------'''
            front = nsga.non_dominated_sorting(self.popsize, chroms_obj_record)

            '''----------selection----------'''
            population, new_pop = nsga.selection(self.popsize, front, chroms_obj_record, total_chromosome)
            new_pop_obj = [chroms_obj_record[k] for k in new_pop]

            '''----------comparison----------'''
            if generation == 0:
                best_list = copy.deepcopy(population)
                best_obj = copy.deepcopy(new_pop_obj)
            else:
                total_list = np.concatenate((copy.deepcopy(population), copy.deepcopy(best_list)))
                total_obj = np.concatenate((copy.deepcopy(new_pop_obj), copy.deepcopy(best_obj)))

                now_best_front = nsga.non_dominated_sorting(self.popsize, total_obj)
                best_list, best_pop = nsga.selection(self.popsize, now_best_front, total_obj, total_list)
                best_obj = [total_obj[k] for k in best_pop]


def main():
    # Các thông số của thuật toán
    popsize = 50
    num_parents_mating = 20
    num_generations = 20,
    distribution_index = 50
    HC_penalt_point = 200
    SC_penalt_point = 3

    # Tạo một đối tượng PMSBX_GA_Algorithm
    pmsbx_ga_algorithm = PMSBX_GA_Algorithm(popsize, num_parents_mating, num_generations, distribution_index,
                                            HC_penalt_point, SC_penalt_point)

    # Chạy thuật toán
    pmsbx_ga_algorithm.run_algorithm()


if __name__ == "__main__":
    main()
