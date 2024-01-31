import datetime
import copy
import pandas as pd
import random

from chromosome import *


"""Create population function - PMSBX-GA"""


def init_population(size_of_population):
    population = []
    for i in range(size_of_population):
        individual = CHROMOSOME_PMSBX_GA(get_data())
        population.append(individual)
    population = np.asarray(population)
    return population


"""Select a sub-population for offspring production - PMSBX-GA"""


def select_mating_pool(pop, num_parents_mating):
    # shuffling the pop then select top of pops
    pop = np.asarray(pop)
    index = np.random.choice(pop.shape[0], num_parents_mating, replace=False)
    random_individual = pop[index]
    # split current pop into remain_pop and mating_pool
    # pop = np.delete(pop, index)
    return random_individual

"""Crossover - PMSBX-GA"""


def crossover(parents, distribution_index):
    offspring = []

    for k in range(0,parents.shape[0],2):
        # Index of the first parent to mate.
        parent1_idx = k % parents.shape[0]
        # Index of the second parent to mate.
        if k + 1 >= len(parents):
            break
        parent2_idx = (k + 1) % parents.shape[0]
        parent1 = parents[parent1_idx]
        parent2 = parents[parent2_idx]
        chromosome_1 = parent1.chromosome
        chromosome_2 = parent2.chromosome
        offspring_1= []
        offspring_2= []
        for task1,task2 in zip(chromosome_1,chromosome_2):
            gen_1 = parser_gen_pmsbx(task1)
            gen_2 = parser_gen_pmsbx(task2)
            new_gen_1, new_gen_2 = crossover_calculation(gen_1,gen_2, distribution_index)
            offspring_1.append(new_gen_1)
            offspring_2.append(new_gen_2)

        offspring.append(offspring_1)
        offspring.append(offspring_2)
    return np.array(offspring)

"""Mutation - PMSBX-GA"""


def mutation(offspring, distribution_index):
    new_offspring = []
    for index in range(len(offspring)):
        chromosome = offspring[index]
        offspring_temp= []
        for gen in range(len(chromosome)):
            new_gen = mutation_calculation(chromosome[gen], distribution_index)
            offspring_temp.append(new_gen)
        new_offspring.append(offspring_temp)
    return np.array(new_offspring)

def crossover_calculation(gen_1,gen_2, distribution_index):
    # Khoi tao bien group selected variables
    # Gen_structure = (id, start_date, end_date, scheduled_date, routine, battery_type, num_battery)
    # v = (routine,(scheduled_date − start_date), battery_type, num_battery)
    beta_para = 0

    diff_date_gen_1 = difference_date(gen_1.scheduled_date, gen_1.start_date)
    diff_date_gen_2 = difference_date(gen_1.scheduled_date, gen_1.start_date)

    candidate_1 = (0,0,0,0)
    candidate_2 = (0,0,0,0)

    check = False
    while check == False:
        random_rate = generate_random_number()
        if random_rate <= 0.5:
            beta_para = power_of_fraction(2*random_rate, 1,distribution_index +1)
        elif random_rate > 0.5 :
            beta_para = power_of_fraction((2-2*random_rate), 1,distribution_index +1)

        vector_v1 = (int(gen_1.routine), diff_date_gen_1.days, int(gen_1.battery_type), int(gen_1.num_battery))
        vector_v2 = (int(gen_2.routine), diff_date_gen_2.days, int(gen_2.battery_type), int(gen_2.num_battery))

        #v1_new = 0.5 × [(1 + β)υ1 + (1 − β)υ2]
        candidate_1 = scalar_multiply_v1_crossover(beta_para, vector_v1, vector_v2)
        #v2_new = 0.5 × [(1 - β)υ1 + (1 + β)υ2]
        candidate_2 = scalar_multiply_v2_crossover(beta_para, vector_v1, vector_v2)

        # Check for violations of each variable in v_1 and v_2
        check = check_violations(candidate_1, candidate_2)
    return candidate_1, candidate_2


def check_violations(candidate_1, candidate_2):
    #['routine', 'difference_date', 'battery_type', 'num_battery']

    # check  routine ∈ {0, 1} is the ROUTINE variable.
    if candidate_1.routine <= 0 and candidate_1.routine >= 1:
        return False
    if candidate_2.routine <= 0 and candidate_2.routine >= 1:
        return False

    #difference_date ∈ [0..30] is the difference between SCHEDULED DATE and the START DATE.
    if candidate_1.difference_date <= 0 and candidate_1.difference_date >= 30:
        return False
    if candidate_2.difference_date <= 0 and candidate_2.difference_date >= 30:
        return False

    # battery_type ∈ [1..5] is the BATTERY TYPE variable
    if candidate_1.battery_type <= 1 and candidate_1.battery_type >= 5:
        return False
    if candidate_2.battery_type <= 1 and candidate_2.battery_type >= 5:
        return False

    #num_battery ∈ [1..10] is the NUMBER OF BATTERIES variable
    if candidate_1.num_battery <= 1 and candidate_1.num_battery >= 10:
        return False
    if candidate_2.num_battery <= 1 and candidate_2.num_battery >= 10:
        return False
    return True


def mutation_calculation(parents, distribution_index):
    # Khoi tao bien group selected variables
    # v = (routine,(scheduled_date − start_date), battery_type, num_battery)
    delta_para = 0
    new_gen = (0,0,0,0)
    # Generate a random number ε ∈ R, where 0 ≤ ε ≤ 1
    random_rate = generate_random_number()
    if random_rate <= 0.5:
        delta_para = power_of_fraction(2*random_rate, 1,distribution_index +1) - 1
    else :
        delta_para = 1 - power_of_fraction((2-2*random_rate), 1,distribution_index +1)
    temp = Vector(*parents)
    vector = (temp.routine, temp.difference_date, temp.battery_type, temp.num_battery)

    new_gen = scalar_multiply_motation(vector, delta_para, random_rate)
    return new_gen

"""Selecting a new population for the next generation from parents and offsprings - PMSBX-GA"""


def selection(parents, offsprings):
    pass
