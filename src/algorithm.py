import copy

from src.NSGA import nsga
from src.PMSBX_GA import pmsbx_ga
from src.GA import ga
import numpy as np
import pandas as pd
import statistics
import sys
import random
from src.NSGA.conditions_nsga import *

"""Create chromosome - NSGA"""

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

    def print_initial_info(self):
        print(f"Starting NSGA-II with population size: {self.popsize}")
        print(f"Number of generations: {self.num_generations}")
        print(f"Number of parents mating: {self.num_parents_mating}")
        print(f"Mutation rate: {self.mutation_rate}")
        print(f"HC penalty point: {self.HC_penalt_point}, SC penalty point: {self.SC_penalt_point}")
        print("=" * 50)

    def print_population_fitness(self, population):
        print("Population fitness:")
        for i, individual in enumerate(population):
            HC_time, HC_resource = nsga.fitness_value(individual)
            print(f"Individual {i}: HC_time = {HC_time}, HC_resource = {HC_resource}")

    def print_best_individual(self, best_individual):
        HC_time, HC_resource = nsga.fitness_value(best_individual)
        print(f"Best individual: HC_time = {HC_time}, HC_resource = {HC_resource}")

    def print_generation_update(self, old_population, new_population):
        print(f"Updating generation from {len(old_population)} to {len(new_population)} individuals")

    def run_algorithm(self):
        self.print_initial_info()
        population = nsga.init_population(self.popsize)

        for generation in range(self.num_generations):
            print(f"\nGeneration {generation + 1}/{self.num_generations}")

            parents = nsga.select_mating_pool(population, self.num_parents_mating)
            offspring_crossover = nsga.crossover(parents)
            offspring_mutation = nsga.mutation(population, self.mutation_rate)
            chroms_obj_record = {}  # record each chromosome objective values as chromosome_obj_record={chromosome:[HC_time,HC_record]}
            # Kết hợp quần thể hiện tại và con cái để tạo ra một quần thể mới
            total_chromosome = np.concatenate(
                (copy.deepcopy(parents), copy.deepcopy(offspring_crossover), copy.deepcopy(offspring_mutation)))

            chroms_obj_record = {}
            for m in range(len(total_chromosome)):
                HC_time, HC_resource = nsga.cal_hc_time_and_resource(total_chromosome[m])
                # HC_time, HC_resource = nsga.fitness_value(total_chromosome[m])
                # fitness_values = cal_fitness_value(population, self.HC_penalt_point, self.SC_penalt_point)
                chroms_obj_record[m] = [HC_time, HC_resource]
            self.print_population_fitness(total_chromosome)

            '''-------non-dominated sorting-------'''
            front = nsga.non_dominated_sorting(self.popsize, chroms_obj_record)

            '''----------selection----------'''
            population, new_pop = nsga.selection(self.popsize, front, chroms_obj_record, total_chromosome)

            self.print_generation_update(total_chromosome, population)

            new_pop_obj = [chroms_obj_record[k] for k in new_pop]

            '''----------comparison and updating the best found so far----------'''
            if generation == 0 or len(best_list) == 0:
                best_list = copy.deepcopy(population)
                best_obj = copy.deepcopy(new_pop_obj)
            else:
                total_list = np.concatenate((copy.deepcopy(population), copy.deepcopy(best_list)))
                total_obj = np.concatenate((copy.deepcopy(new_pop_obj), copy.deepcopy(best_obj)))

                now_best_front = nsga.non_dominated_sorting(self.popsize, total_obj)
                best_list, best_pop = nsga.selection(self.popsize, now_best_front, total_obj, total_list)
                best_obj = [total_obj[k] for k in best_pop]

            # Tìm và in cá thể tốt nhất trong quần thể
            if best_list:
                self.print_best_individual(best_list[0])

    # Lưu ý: Mã giả định rằng các hàm `init_population`, `select_mating_pool`, `crossover`, `mutation`, `fitness_value`, `non_dominated_sorting`, và `selection` đã được định nghĩa và hoạt động đúng đắn.


def main():
    # Các thông số của thuật toán
    popsize = 6
    num_parents_mating = 4
    num_generations = 10
    distribution_index = 100
    HC_penalt_point = 10
    SC_penalt_point = 3
    #
    nsga_algorithm = NSGA_ALGORITHM(
        popsize,
        num_parents_mating,
        num_generations,
        0.3,
        HC_penalt_point,
        SC_penalt_point,
    )
    nsga_algorithm.run_algorithm();

    # pmsbx_ga_algorithm = PMSBX_GA_Algorithm(
    #     popsize,
    #     num_parents_mating,
    #     num_generations,
    #     distribution_index,
    #     HC_penalt_point,
    #     SC_penalt_point,
    # )
    # # Chạy thuật toán
    # pmsbx_ga_algorithm.run_algorithm()
if __name__ == "__main__":
    main()
    # Tạo một đối tượng PMSBX_GA_Algorithm

