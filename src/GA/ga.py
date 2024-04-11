import datetime
import copy
import pandas as pd
import random

from src.chromosome import *
from src.PMSBX_GA.pmsbx_ga import _generate_parent
from src.utilities.utils import *
from src.GA.conditions_ga import cal_fitness_value
from src.utilities.draw_chart import draw_fitness

# from datetime import datetime

"""Create population function - GA"""


def _generate_parent(self):
    genes = []
    # for supply_id, start_day, end_date, battery_type in zip(
    #     self.df.supply_id,
    #     self.df.start_day,
    #     self.df.end_date,
    #     self.df.battery_type,
    # ):
    for supply_id, start_day, end_date, battery_type, d_estdur in zip(
        self.supply_id, self.start_day, self.end_date, self.battery_type, self.d_estdur
    ):
        rand_date = random_date_bit(start_day, end_date, random.random())
        routine = random.choice([0, 1])
        battery_type = battery_type.split("|")
        battery_type_gen = random.choice(battery_type)
        battery_type_gen = battery_type_bit[battery_type_gen]

        random_num_battery = random.randint(0, 10)
        # Chuyển số nguyên sang số nhị phân
        num_battery = format(random_num_battery, "04b")
        bitstring = "".join([rand_date, str(routine), battery_type_gen, num_battery])
        chromosome = "-".join([supply_id, start_day, end_date, bitstring])
        genes.append(chromosome)
    return np.asarray(genes)


def init_population(size_of_population):
    population = []
    for i in range(size_of_population):
        individual = _generate_parent(get_data())
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


import numpy as np


# def crossover(parents):
#     # Đảm bảo rằng parents là một mảng 2D
#     if parents.ndim == 1:
#         parents = np.expand_dims(parents, axis=0)

#     # Initializes a numpy array offspring by copying the first half of the parents array.
#     offspring = np.copy(parents[: parents.shape[0] // 2])

#     # The point at which crossover takes place between two parents, which in this case is at the center.
#     crossover_point = parents.shape[1] // 2

#     # Perform crossover
#     for k in range(offspring.shape[0]):
#         # Index of the first parent to mate.
#         parent1_idx = k % parents.shape[0]
#         # Index of the second parent to mate.
#         parent2_idx = (k + 1) % parents.shape[0]

#         # The new offspring will have its first half of its genes taken from the first parent.
#         offspring[k, :crossover_point] = parents[parent1_idx, :crossover_point]
#         # The new offspring will have its second half of its genes taken from the second parent.
#         offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]

#     return offspring


def crossover(parents):
    mating_pool = copy.deepcopy(parents)
    offsprings = []
    while mating_pool.size > 0:
        mating_idx = np.random.choice(mating_pool.shape[0], 2, replace=False)
        mating_parents = mating_pool[mating_idx]
        parent_1 = mating_parents[0]
        parent_2 = mating_parents[1]
        swap_task_pos = random.randrange(parent_1.shape[0])
        # print("parent_1: ", parent_1[swap_task_pos])
        # Tính toán phạm vi cho hàm random.sample()
        # start_index = parent_1[0].rfind("-") + 1  # Vị trí của ký tự '-' đầu tiên
        # end_index = len(parent_1[0]) - 1  # Độ dài của chuỗi - 1

        # # Tạo danh sách các index không thuộc vào khoảng [13:16] (không đổi loại pin)
        # valid_indexes = list(range(start_index, start_index + 12)) + list(
        #     range(start_index + 16, end_index + 1)
        # )

        # # Chọn ngẫu nhiên 2 điểm cắt từ danh sách các index hợp lệ
        # crossover_point = random.sample(valid_indexes, 2)
        crossover_point = random.sample(
            range(parent_1[0].rfind("-") + 1, len(parent_1[0])),
            2,
        )
        crossover_point.sort()
        # print(crossover_point)
        offspring_1 = (
            parent_1[swap_task_pos][0 : crossover_point[0]]
            + parent_2[swap_task_pos][crossover_point[0] : crossover_point[1]]
            + parent_1[swap_task_pos][crossover_point[1] :]
        )
        offspring_2 = (
            parent_2[swap_task_pos][0 : crossover_point[0]]
            + parent_1[swap_task_pos][crossover_point[0] : crossover_point[1]]
            + parent_2[swap_task_pos][crossover_point[1] :]
        )
        parent_1[swap_task_pos] = offspring_1
        parent_2[swap_task_pos] = offspring_2
        offsprings.append(parent_1)
        offsprings.append(parent_2)
        mating_pool = np.delete(mating_pool, list(mating_idx), axis=0)

    return np.array(offsprings)


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


def selection(
    parents, offsprings, HC_penalt_point, SC_penalt_point
):  # num individual = num parents
    # Combine parents and offsprings
    if parents.ndim == 1:
        parents = np.expand_dims(parents, axis=0)

        # Ensure offsprings is 2D
    if offsprings.ndim == 1:
        offsprings = np.expand_dims(offsprings, axis=0)

        # Now, concatenate should work as both arrays are 2D
    population = np.concatenate((parents, offsprings), axis=0)

    # Calculate fitness for each individual in the population
    fitness_array = cal_fitness_value(population, HC_penalt_point, SC_penalt_point)

    # Select the best individuals for the next generation
    num_parents = parents.shape[0]  # parents.shape[0] = num_parents_mating
    new_population = population[fitness_array.argsort()[-num_parents:]]  # first n-largest fitness
    return new_population