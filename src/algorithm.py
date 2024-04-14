import copy
import matplotlib.pyplot as plt
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
        self.fig, self.ax = plt.subplots()
        plt.ion()

    def update_plot(self, objectives, generation):
        # Clear the previous scatter plot
        self.ax.clear()

        # Extract HC_time and HC_resource for plotting
        HC_times = [obj[0] for obj in objectives]
        HC_resources = [obj[1] for obj in objectives]

        # Create a scatter plot
        self.ax.scatter(HC_times, HC_resources, label=f'Generation {generation}')

        # Set the labels and title again since the plot is cleared.
        self.ax.set_title('NSGA-II Objectives')
        self.ax.set_xlabel('HC_time')
        self.ax.set_ylabel('HC_resource')
        self.ax.legend()

        # Draw the plot and pause for a brief moment to update the display
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.1)
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
            HC_time, HC_resource = nsga.cal_hc_time_and_resource(individual)
            print(f"Individual {i}: HC_time = {HC_time}, HC_resource = {HC_resource}")

    def print_best_individual(self, best_individual):
        HC_time, HC_resource = nsga.cal_hc_time_and_resource(best_individual)
        print(f"Best individual: HC_time = {HC_time}, HC_resource = {HC_resource}")

    def print_generation_update(self, old_population, new_population):
        print(f"Updating generation from {len(old_population)} to {len(new_population)} individuals")

    def plot_objectives(self, objectives, generation):
        HC_times = [obj[0] for obj in objectives]
        HC_resources = [obj[1] for obj in objectives]

        plt.scatter(HC_times, HC_resources, label=f'Generation {generation}')
        plt.title('NSGA-II Objectives')
        plt.xlabel('HC_time')
        plt.ylabel('HC_resource')
        plt.legend()
        plt.show()
    # def plot_fitness(self, fitness_values):
    #     # Generate a line plot of fitness values over generations
    #     plt.plot(fitness_values, '-o', label='Fitness over Generations')
    #     plt.title('Fitness over Generations')
    #     plt.xlabel('Generation')
    #     plt.ylabel('Fitness')
    #     plt.legend()
    #     plt.grid(True)
    #     plt.show()
    def calculate_fitness(self, HC_time, HC_resource):
        # Prevent division by zero in case both HC_time and HC_resource are zero
        return 1 / (HC_time + HC_resource + 1e-6)

    def run_algorithm(self):
        self.print_initial_info()
        population = nsga.init_population(self.popsize)
        best_fitness_over_gens = []  # Store the best fitness for each generation

        for generation in range(self.num_generations):
            print(f"\nGeneration {generation + 1}/{self.num_generations}")

            parents = nsga.select_mating_pool(population, self.num_parents_mating)
            offspring_crossover = nsga.crossover(parents)
            offspring_mutation = nsga.mutation(population, self.mutation_rate)
            chroms_obj_record = {}
            total_chromosome = np.concatenate((copy.deepcopy(parents), copy.deepcopy(
                offspring_crossover), copy.deepcopy(population), copy.deepcopy(
                offspring_mutation)))  # combine parent and offspring chromosomes
            for m in range(self.popsize * 2):
                HC_time, HC_resource = nsga.cal_hc_time_and_resource(total_chromosome[m])
                chroms_obj_record[m] = [HC_time, HC_resource]
            # for m in range(self.popsize * 2):
            #     HC_time, HC_resource = nsga.cal_hc_time_and_resource(total_chromosome[m])
            #     chroms_obj_record[m] = [HC_time, HC_resource]

            '''-------non-dominated sorting-------'''
            front = nsga.non_dominated_sorting(self.popsize, chroms_obj_record)

            '''----------selection----------'''
            population, new_pop = nsga.selection(self.popsize, front, chroms_obj_record, total_chromosome)

            new_pop_obj = [chroms_obj_record[k] for k in new_pop]

            # Calculate fitness with list handling
            fitness_values = []
            for obj in new_pop_obj:
                # Extract HC_time and HC_resource assuming they might be wrapped in lists
                HC_time = obj[0][0] if isinstance(obj[0], list) else obj[0]
                HC_resource = obj[1][0] if isinstance(obj[1], list) else obj[1]

                # Check if extracted values are numbers
                if isinstance(HC_time, (float, int)) and isinstance(HC_resource, (float, int)):
                    fitness = 1 / (HC_time + HC_resource + 1e-6)
                    fitness_values.append(fitness)
                else:
                    print(f"Error: Extracted values are not numbers. HC_time: {HC_time}, HC_resource: {HC_resource}")

            if fitness_values:
                best_individual_fitness = min(fitness_values)
                best_fitness_over_gens.append(best_individual_fitness)
            else:
                print("Warning: No valid fitness values were calculated for this generation.")

            # Plot the fitness of the best individual over generations
        self.plot_fitness(best_fitness_over_gens)

    def plot_fitness(self, fitness_values):
        # Generate a line plot of fitness values over generations
        plt.figure(figsize=(10, 5))  # Optional: specify figure size
        plt.plot(fitness_values, '-o', label='Fitness over Generations')
        plt.title('Best Fitness over Generations')
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.legend()
        plt.grid(True)
        plt.show()


    # Lưu ý: Mã giả định rằng các hàm `init_population`, `select_mating_pool`, `crossover`, `mutation`, `fitness_value`, `non_dominated_sorting`, và `selection` đã được định nghĩa và hoạt động đúng đắn.

# class PMSBX_NSGA_Algorithm:
#
def main():
    # Các thông số của thuật toán
    popsize = 40
    num_parents_mating = 20
    num_generations = 20
    # distribution_index = 100
    mutation_rate = 0.3
    HC_penalt_point = 10
    SC_penalt_point = 3
    #
    nsga_algorithm = NSGA_ALGORITHM(
        popsize,
        num_parents_mating,
        num_generations,
        mutation_rate,
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

