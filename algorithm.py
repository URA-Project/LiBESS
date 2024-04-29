import copy
import matplotlib.pyplot as plt
from NSGA import nsga
from PMSBX_GA import pmsbx_ga
from GA import ga
import numpy as np
import pandas as pd
import statistics
import sys
import random
from PMSBX_NSGA import pmsbx_nsga

import sys

sys.path.insert(
    0,
    "/Users/mac/Library/CloudStorage/OneDrive-Personal/Study/URA/GA_Emerging_Papers/LiBESS-MAIN/src",
)

"""Create chromosome - NSGA"""


class GA_ALGORITHM:
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
        best_fitness_over_gens = []  # Store the best fitness for each generation
        fitness = []
        data_draw = []
        # Add code evalue fitnesss
        for index in range(self.num_generations):
            print("-------Lần thứ: ", index)
            print_fitness = []
            # Selecting the best parents in the population for mating.
            parents = ga.select_mating_pool(population, self.num_parents_mating)
            # Crossover
            offspring_crossover = ga.crossover(parents)
            # Take every individual in the offspring after crossover to mutate with a given rate
            offspring_mutation = ga.mutation(offspring_crossover, self.mutation_rate)
            total_chromosome = np.concatenate(
                (
                    copy.deepcopy(offspring_crossover),
                    copy.deepcopy(population),
                    copy.deepcopy(offspring_mutation),
                )
            )  # combine parent and offspring chromosomes
            chroms_obj_record = {}
            for m in range(len(total_chromosome)):
                total_capacity, resource_and_deadline = ga.cal_hc_time_and_resource(
                    total_chromosome[m]
                )
                chroms_obj_record[m] = [resource_and_deadline]
                print_fitness.append(resource_and_deadline)

            # Selecting a new population for the next generation from parents and offsprings
            population_list, new_chroms_obj_record = ga.selection(
                self.popsize, chroms_obj_record, total_chromosome
            )
            population = population_list
            print("new_chroms_obj_record: ", new_chroms_obj_record)
            best_fitness_over_gens.append(new_chroms_obj_record[0])
            ga.save_output(population_list[0], index)

        print("best_fitness_over_gens: ", best_fitness_over_gens)


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

    def run_algorithm(self):
        population = nsga.init_population(self.popsize)
        best_fitness_over_gens = []  # Store the best fitness for each generation
        fitness = []
        fitness_values = []
        for index in range(self.num_generations):
            print_fitness = []
            print("------- Lan lap thu: ", index + 1)
            parents = nsga.select_mating_pool(population, self.num_parents_mating)
            offspring_crossover = nsga.crossover(parents)
            offspring_mutation = nsga.mutation(population, self.mutation_rate)
            chroms_obj_record = {}
            total_chromosome = np.concatenate(
                (
                    copy.deepcopy(offspring_crossover),
                    copy.deepcopy(population),
                    copy.deepcopy(offspring_mutation),
                )
            )  # combine parent and offspring chromosomes
            unique_total_chromosome = []
            lengh = len(total_chromosome)
            i = 0
            for m in range(lengh):
                HC_time, HC_resource = nsga.cal_hc_time_and_resource(
                    total_chromosome[m]
                )
                if [HC_time, HC_resource] not in chroms_obj_record.values():
                    chroms_obj_record[i] = [HC_time, HC_resource]
                    unique_total_chromosome.append(total_chromosome[m])
                    i = i + 1
                    print_fitness.append(HC_resource + HC_time)

            """-------non-dominated sorting-------"""
            front, points = nsga.non_dominated_sorting(
                len(unique_total_chromosome), chroms_obj_record
            )

            """----------selection----------"""
            population, new_pop = nsga.selection(
                self.popsize, front, chroms_obj_record, unique_total_chromosome
            )

            """----------Xuat gia tri output----------"""
            best_fitness = {}
            for point_index in front[0]:
                value = chroms_obj_record[point_index]
                best_fitness[point_index] = value[1]

            # Lấy chỉ mục của giá trị nhỏ nhất
            min_index = min(best_fitness, key=best_fitness.get)
            nsga.save_output(unique_total_chromosome[min_index], index)

            # Calculate fitness with list handling
            fitness_values.append(
                chroms_obj_record[min_index][0] + chroms_obj_record[min_index][1]
            )

        """----------Lưu các best fitness ra file----------"""
        best_fitness_path = "/Users/mac/Library/CloudStorage/OneDrive-Personal/Study/URA/GA_Emerging_Papers/LiBESS-MAIN/output/best_fitness_nsga.txt"
        with open(best_fitness_path, "w") as file:
            for fitness in fitness_values:
                file.write(str(fitness) + "\n")
        print("best_fitness_over_gens: ", fitness_values)


class PMSBX_NSGA_ALGORITHM:
    def __init__(
        self,
        popsize,
        num_parents_mating,
        num_generations,
        HC_penalt_point,
        SC_penalt_point,
        distribution_index,
    ):
        self.popsize = popsize
        self.num_parents_mating = num_parents_mating
        self.num_generations = num_generations
        self.HC_penalt_point = HC_penalt_point
        self.SC_penalt_point = SC_penalt_point
        self.distribution_index = distribution_index

    def run_algorithm(self):
        population = pmsbx_nsga.init_population(self.popsize)
        best_fitness_over_gens = []  # Store the best fitness for each generation
        for index in range(self.num_generations):
            print("-------Lần thứ: ", index + 1)
            parents = pmsbx_nsga.select_mating_pool(population, self.num_parents_mating)
            offspring_crossover = pmsbx_nsga.crossover(parents, self.distribution_index)
            offspring_mutation = pmsbx_nsga.mutation(
                population, self.distribution_index
            )
            chroms_obj_record = {}

            total_chromosome = np.concatenate(
                (
                    copy.deepcopy(offspring_crossover),
                    copy.deepcopy(offspring_mutation),
                    copy.deepcopy(population),
                )
            )  # combine parent and offspring chromosomes
            for m in range(len(total_chromosome)):
                HC_time, HC_resource = pmsbx_nsga.cal_hc_time_and_resource(
                    total_chromosome[m]
                )
                chroms_obj_record[m] = [HC_time, HC_resource]

            record_copy = chroms_obj_record.copy()
            # Tạo một từ điển mới để lưu trữ các giá trị không trùng lặp
            unique_chroms_obj_record = {}
            unique_total_chromosome = []

            i = 0
            for key, value in record_copy.items():
                # Chỉ thêm vào từ điển mới nếu giá trị không trùng lặp
                if value not in unique_chroms_obj_record.values():
                    unique_chroms_obj_record[i] = value
                    unique_total_chromosome.append(total_chromosome[key])
                    i = i + 1

            """-------non-dominated sorting-------"""
            front, points = pmsbx_nsga.non_dominated_sorting(
                len(unique_total_chromosome), chroms_obj_record
            )

            """----------selection----------"""
            population, new_pop = pmsbx_nsga.selection(
                self.popsize, front, chroms_obj_record, total_chromosome
            )

            best_fitness = {}
            for point_index in front[0]:
                value = chroms_obj_record[point_index]
                best_fitness[point_index] = value[1]

            # Lấy chỉ mục của giá trị nhỏ nhất
            min_index = min(best_fitness, key=best_fitness.get)
            pmsbx_nsga.save_output(unique_total_chromosome[min_index], index)

            # Calculate fitness with list handling
            best_fitness_over_gens.append(
                chroms_obj_record[min_index][0] + chroms_obj_record[min_index][1]
            )

        """----------Lưu các best fitness ra file----------"""
        best_fitness_path = "/Users/mac/Library/CloudStorage/OneDrive-Personal/Study/URA/GA_Emerging_Papers/LiBESS-MAIN/output/best_fitness_pmspx_nsga.txt"
        with open(best_fitness_path, "w") as file:
            for fitness in best_fitness_over_gens:
                file.write(str(fitness) + "\n")
        print("best_fitness_over_gens: ", best_fitness_over_gens)


def main_pmsbx_nsga():
    # Các thông số của thuật toán
    popsize = 6
    num_parents_mating = 4
    num_generations = 20
    distribution_index = 80
    HC_penalt_point = 10
    SC_penalt_point = 3

    pmsbx_nsga_algorithm = PMSBX_NSGA_ALGORITHM(
        popsize,
        num_parents_mating,
        num_generations,
        HC_penalt_point,
        SC_penalt_point,
        distribution_index,
    )
    pmsbx_nsga_algorithm.run_algorithm()


def main_nsga():
    # Các thông số của thuật toán
    popsize = 6
    num_parents_mating = 4
    num_generations = 20
    mutation_rate = 0.5
    HC_penalt_point = 10
    SC_penalt_point = 3
    nsga_algorithm = NSGA_ALGORITHM(
        popsize,
        num_parents_mating,
        num_generations,
        mutation_rate,
        HC_penalt_point,
        SC_penalt_point,
    )
    nsga_algorithm.run_algorithm()


def main_ga():
    # Các thông số của thuật toán
    popsize = 6
    num_parents_mating = 4
    num_generations = 30
    mutation_rate = 0.5
    HC_penalt_point = 10
    SC_penalt_point = 3
    ga_algorithm = GA_ALGORITHM(
        popsize,
        num_parents_mating,
        num_generations,
        mutation_rate,
        HC_penalt_point,
        SC_penalt_point,
    )
    ga_algorithm.run_algorithm()


if __name__ == "__main__":
    # main_ga()
    # main_nsga()
    main_pmsbx_nsga()
