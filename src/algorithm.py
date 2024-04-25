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
sys.path.insert(0, '/LiBESS-NSGA-II-V2/src')

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

    def run_algorithm(self):
        # self.print_initial_info()
        population = nsga.init_population(self.popsize)
        best_fitness_over_gens = []  # Store the best fitness for each generation
        fitness = []
        data_draw = []
        for index in range(self.num_generations):
            print_fitness = []
            # print(f"\nGeneration {generation + 1}/{self.num_generations}")

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
                print_fitness.append(HC_resource + HC_time)
            
            abc = sorted(print_fitness)
            print("print_fitness", abc)
            # for m in range(self.popsize * 2):
            #     HC_time, HC_resource = nsga.cal_hc_time_and_resource(total_chromosome[m])
            #     chroms_obj_record[m] = [HC_time, HC_resource]

            '''-------non-dominated sorting-------'''
            front, points = nsga.non_dominated_sorting(self.popsize, chroms_obj_record)

            '''----------lưu các điểm pareto----------'''
            if (index + 1) % 20 == 0 or index == 0:
                for key, value in points.items():
                    points[key] = [float(val) for val in value]
                all_points = []
                for point_index in front[0]:
                    all_points.append(point_index)

                # # Tạo danh sách các điểm trong Pareto Front
                pareto_points = [points[index] for index in all_points]
                data_draw.append(pareto_points)

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
                    # fitness = 1 / (HC_time + HC_resource + 1e-6)
                    fitness = HC_time + HC_resource
                    fitness_values.append(fitness)
                else:
                    print(f"Error: Extracted values are not numbers. HC_time: {HC_time}, HC_resource: {HC_resource}")

            if fitness_values:
                best_individual_fitness = min(fitness_values)
                best_fitness_over_gens.append(best_individual_fitness)
            else:
                print("Warning: No valid fitness values were calculated for this generation.")


        '''----------Lưu các đường pareto ra file và vẽ hình ----------'''
        #Nên tạo fuction và di chuyển code lưu file ra ngoài hàm riêng
        file_path = "/LiBESS-NSGA-II-V2/data/data_draw.txt"
        with open(file_path, 'w') as file:
            for pareto_points in data_draw:
                file.write("-------------------")
                for point in pareto_points:
                    file.write(f"({point[0]}, {point[1]})\n")

        # Danh sách các màu
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'yellow', 'cyan', 'magenta']
        # Vẽ từng pareto_points trong data_draw với màu khác nhau
        count = 0
        for i, pareto_points in enumerate(data_draw):
            if i==0:
                # Tách các điểm Pareto thành 2 list để vẽ
                pareto_y = [p[0] for p in pareto_points]
                pareto_x = [p[1] for p in pareto_points]
                # Chọn màu từ danh sách colors (lặp lại nếu hết)
                color = colors[i % len(colors)]
                # Vẽ đường Pareto
                plt.scatter(pareto_x, pareto_y, color=color, label=f'Lần lặp thứ: {i+1}')
            else:
                # Tách các điểm Pareto thành 2 list để vẽ
                pareto_y = [p[0] for p in pareto_points]
                pareto_x = [p[1] for p in pareto_points]
                # Chọn màu từ danh sách colors (lặp lại nếu hết)
                color = colors[i % len(colors)]
                # Vẽ đường Pareto
                plt.scatter(pareto_x, pareto_y, color=color, label=f'Lần lặp thứ: {20 + count}')
                count = count + 20
        # Đặt các thông số trục và tiêu đề
        plt.xlabel('Resource')
        plt.ylabel('Total capacity')
        plt.title('Pareto Front')
        # Hiển thị chú thích (legend)
        plt.legend()
        # Hiển thị đồ thị
        # Lưu hình ảnh
        plt.savefig('/LiBESS-NSGA-II-V2/data/pareto_fronts.png')
        plt.show()
        # Store the best fitness for each generation
        print("best_fitness_over_gens", best_fitness_over_gens)

        '''----------Lưu các best fitness ra file----------'''
        best_fitness = "/LiBESS-NSGA-II-V2/data/best_fitness.txt"
        with open(best_fitness, 'w') as file:
            for fitness in best_fitness_over_gens:
                file.write(str(fitness) + '\n')

class PMSBX_NSGA_ALGORITHM:
    def __init__(
            self,
            popsize,
            num_parents_mating,
            num_generations,
            mutation_rate,
            HC_penalt_point,
            SC_penalt_point,
            distribution_index,
    ):
        # Need to update
        self.popsize = popsize
        self.num_parents_mating = num_parents_mating
        self.num_generations = num_generations
        self.mutation_rate = mutation_rate
        self.best_outputs = []
        self.HC_penalt_point = HC_penalt_point
        self.SC_penalt_point = SC_penalt_point
        self.distribution_index = distribution_index
        self.fig, self.ax = plt.subplots()
        plt.ion()

    def run_algorithm(self):
        population = pmsbx_nsga.init_population(self.popsize)
        best_fitness_over_gens = []  # Store the best fitness for each generation

        data_draw = []
        for index in range(self.num_generations):
            print_fitness = []
            parents = pmsbx_nsga.select_mating_pool(population, self.num_parents_mating)
            offspring_crossover = pmsbx_nsga.crossover(parents, self.distribution_index)
            offspring_mutation = pmsbx_nsga.mutation(population, self.distribution_index)
            chroms_obj_record = {}

            total_chromosome = np.concatenate((copy.deepcopy(parents), copy.deepcopy(
                offspring_crossover), copy.deepcopy(population), copy.deepcopy(
                offspring_mutation)))  # combine parent and offspring chromosomes
            for m in range(self.popsize * 2):
                HC_time, HC_resource = pmsbx_nsga.cal_hc_time_and_resource(total_chromosome[m])
                print_fitness.append(HC_resource + HC_time)
                chroms_obj_record[m] = [HC_time, HC_resource]

            '''-------non-dominated sorting-------'''
            front, points = pmsbx_nsga.non_dominated_sorting(self.popsize, chroms_obj_record)

            '''----------Lưu các điểm pareto----------'''
            if (index + 1) % 2 == 0 or index == 0:
                for key, value in points.items():
                    points[key] = [float(val) for val in value]
                all_points = []
                for point_index in front[0]:
                    all_points.append(point_index)
                # # Tạo danh sách các điểm trong Pareto Front
                pareto_points = [points[index] for index in all_points]
                data_draw.append(pareto_points)

            '''----------selection----------'''
            population, new_pop = pmsbx_nsga.selection(self.popsize, front, chroms_obj_record, total_chromosome)

            new_pop_obj = [chroms_obj_record[k] for k in new_pop]

            # Calculate fitness with list handling
            fitness_values = []
            for obj in new_pop_obj:
                # Extract HC_time and HC_resource assuming they might be wrapped in lists
                HC_time = obj[0][0] if isinstance(obj[0], list) else obj[0]
                HC_resource = obj[1][0] if isinstance(obj[1], list) else obj[1]

                # Check if extracted values are numbers
                if isinstance(HC_time, (float, int)) and isinstance(HC_resource, (float, int)):
                    # fitness = 1 / (HC_time + HC_resource)
                    fitness = HC_time + HC_resource
                    fitness_values.append(fitness)
                else:
                    print(f"Error: Extracted values are not numbers. HC_time: {HC_time}, HC_resource: {HC_resource}")

            if fitness_values:
                best_individual_fitness = min(fitness_values)
                best_fitness_over_gens.append(best_individual_fitness)
                # print("best_fitness_over_gens: ", best_fitness_over_gens )
            else:
                print("Warning: No valid fitness values were calculated for this generation.")

        '''----------Lưu các đường pareto ra file và vẽ hình ----------'''
        #Nên tạo fuction và di chuyển code lưu file ra ngoài hàm riêng
        file_path = "C:/Users/trong.le-van/OneDrive - Ban Vien Corporation/HCMUT_OneDrive/URA/Paper/LiBESS-NSGA_II/LiBESS-PMSBX-NSGA_II/data/data_draw.txt"
        with open(file_path, 'w') as file:
            for pareto_points in data_draw:
                file.write("-------------------")
                for point in pareto_points:
                    # Ghi từng điểm vào file với định dạng "(x, y)"
                    file.write(f"({point[0]}, {point[1]})\n")
        '''----------Vẽ pareto----------'''
        # Danh sách các màu
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'yellow', 'cyan', 'magenta']
        # Vẽ từng pareto_points trong data_draw với màu khác nhau
        for i, pareto_points in enumerate(data_draw):
            # Tách các điểm Pareto thành 2 list để vẽ
            pareto_y = [p[0] for p in pareto_points]
            pareto_x = [p[1] for p in pareto_points]
            # Chọn màu từ danh sách colors (lặp lại nếu hết)
            color = colors[i % len(colors)]
            # Vẽ đường Pareto
            plt.scatter(pareto_x, pareto_y, color=color, label=f'Pareto Front {i+1}')
        # Đặt các thông số trục và tiêu đề
        plt.xlabel('Resource')
        plt.ylabel('Total capacity')
        plt.title('Pareto Fronts')
        plt.legend()
        plt.savefig('C:/Users/trong.le-van/OneDrive - Ban Vien Corporation/HCMUT_OneDrive/URA/Paper/LiBESS-NSGA_II/LiBESS-PMSBX-NSGA_II/data/pareto_front.png')
        plt.show()
        '''---------- Store the best fitness for each generation-----'''
        best_fitness = "C:/Users/trong.le-van/OneDrive - Ban Vien Corporation/HCMUT_OneDrive/URA/Paper/LiBESS-NSGA_II/LiBESS-PMSBX-NSGA_II/data/best_fitness.txt"
        with open(best_fitness, 'w') as file:
            for fitness in best_fitness_over_gens:
                file.write(str(fitness) + '\n')

def main_pmsbx_nsga():
    # Các thông số của thuật toán
    popsize = 6
    num_parents_mating = 4
    num_generations = 30
    distribution_index = 100
    mutation_rate = 0.3
    HC_penalt_point = 10
    SC_penalt_point = 3

    pmsbx_nsga_algorithm = PMSBX_NSGA_ALGORITHM(
        popsize,
        num_parents_mating,
        num_generations,
        mutation_rate,
        HC_penalt_point,
        SC_penalt_point,
        distribution_index,
    )
    pmsbx_nsga_algorithm.run_algorithm()

def main_nsga():
    # Các thông số của thuật toán
    popsize = 6
    num_parents_mating = 4
    num_generations = 10
    mutation_rate = 0.3
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

if __name__ == "__main__":
    main_pmsbx_nsga()
