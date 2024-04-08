import datetime
import copy
import pandas as pd
import random

from src.chromosome import *
from src.pmsbx_ga import _generate_parent
from src.utils import *
from src.conditions_ga import cal_fitness_value
from src.utils import _cal_end_date

# # bit_dict_team
# # ======================MODIFY======================
# bit_dict_team = {
#     "000": "MECH",
#     "001": "PROD",
#     "010": "E&I",
#     "011": "RES",
#     "100": "RES"
# }
# ======================MODIFY======================
"""Create population function - NSGA"""

def init_population(size_of_population):
    population = []
    for i in range(size_of_population):
        individual = _generate_parent(get_data())
        population.append(individual)
    population = np.asarray(population)
    return population


"""Select a sub-population for offspring production - NSGA"""

def select_mating_pool(pop, num_parents_mating):
    # shuffling the pop then select top of pops
    pop = np.asarray(pop)
    index = np.random.choice(pop.shape[0], num_parents_mating, replace=False)
    random_individual = pop[index]
    pop = np.delete(pop, index)  # split current pop into remain_pop and mating_pool
    return random_individual


"""Crossover - NSGA"""

def crossover(parents):
    mating_pool = copy.deepcopy(parents)
    offsprings = []
    while mating_pool.size > 0:
        mating_idx = np.random.choice(mating_pool.shape[0], 2, replace=False)
        mating_parents = mating_pool[mating_idx]
        parent_1 = mating_parents[0]
        parent_2 = mating_parents[1]
        swap_task_pos = random.randrange(parent_1.chromosome.shape[0])
        crossover_point = random.sample(range(parent_1.chromosome[0].rfind('-') + 1, len(parent_1.chromosome[0]) - 1),
                                        2)
        crossover_point.sort()
        offspring_1 = parent_1.chromosome[swap_task_pos][0:crossover_point[0]] + parent_2.chromosome[swap_task_pos][
                                                                                 crossover_point[0]:crossover_point[
                                                                                     1]] + parent_1.chromosome[
                                                                                               swap_task_pos][
                                                                                           crossover_point[
                                                                                               1]:]
        offspring_2 = parent_2.chromosome[swap_task_pos][0:crossover_point[0]] + parent_1.chromosome[swap_task_pos][
                                                                                 crossover_point[0]:crossover_point[
                                                                                     1]] + parent_2.chromosome[
                                                                                               swap_task_pos][
                                                                                           crossover_point[
                                                                                               1]:]
        parent_1.chromosome[swap_task_pos] = offspring_1
        parent_2.chromosome[swap_task_pos] = offspring_2
        parent_1.HC_time = []
        parent_1.HC_resource = []
        parent_2.HC_time = []
        parent_2.HC_resource = []
        offsprings.append(parent_1)
        offsprings.append(parent_2)
        mating_pool = np.delete(mating_pool, list(mating_idx), axis=0)

    return np.array(offsprings)


"""Mutation - NSGA"""

def mutation(population, random_rate):
    geneSet = ['0', '1']
    pop = copy.deepcopy(population)

    mutation_offspring = []
    # Mutation changes a number of genes as defined by the num_mutations argument. The changes are random.
    for chromosome in pop:
        mutate_flag = 0
        for index, task in enumerate(chromosome.chromosome):
            # ==================MODIFY========================
            if index >= len(chromosome.chromosome) - 3:
                continue
            # ==================MODIFY========================
            rate = random.uniform(0, 1)
            if rate < random_rate:
                index = random.randrange(task.rfind('-') + 1, len(task) - 1)
                newGene, alternate = random.sample(geneSet, 2)
                mutate_gene = alternate \
                    if newGene == task[index] \
                    else newGene
                task = task[:index] + mutate_gene + task[index + 1:]
                if mutate_flag == 0:
                    mutate_flag = 1
        if mutate_flag:
            chromosome.HC_time = []
            chromosome.HC_resource = []
            mutation_offspring.append(chromosome)

    return np.asarray(mutation_offspring)


'''-------non-dominated sorting function-------'''


def non_dominated_sorting(population_size, chroms_obj_record):
    s, n = {}, {}
    front, rank = {}, {}
    front[0] = []
    for p in range(population_size * 2):
        s[p] = []
        n[p] = 0
        for q in range(population_size * 2):
            # TODO: define dominate operator in the if statement
            if ((chroms_obj_record[p][0] < chroms_obj_record[q][0] and chroms_obj_record[p][1] < chroms_obj_record[q][
                1]) or (chroms_obj_record[p][0] <= chroms_obj_record[q][0] and chroms_obj_record[p][1] <
                        chroms_obj_record[q][1])
                    or (chroms_obj_record[p][0] < chroms_obj_record[q][0] and chroms_obj_record[p][1] <=
                        chroms_obj_record[q][1])):
                if q not in s[p]:
                    s[p].append(q)
            elif ((chroms_obj_record[p][0] > chroms_obj_record[q][0] and chroms_obj_record[p][1] > chroms_obj_record[q][
                1]) or (chroms_obj_record[p][0] >= chroms_obj_record[q][0] and chroms_obj_record[p][1] >
                        chroms_obj_record[q][1])
                  or (chroms_obj_record[p][0] > chroms_obj_record[q][0] and chroms_obj_record[p][1] >=
                      chroms_obj_record[q][1])):
                n[p] = n[p] + 1
        if n[p] == 0:
            rank[p] = 0
            if p not in front[0]:
                front[0].append(p)

    i = 0
    while (front[i] != []):
        Q = []
        for p in front[i]:
            for q in s[p]:
                n[q] = n[q] - 1
                if n[q] == 0:
                    rank[q] = i + 1
                    if q not in Q:
                        Q.append(q)
        i = i + 1
        front[i] = Q

    del front[len(front) - 1]
    return front


'''--------calculate crowding distance function---------'''

def calculate_crowding_distance(front, chroms_obj_record):
    distance = {m: 0 for m in front}
    # o is objective function's values, i.e: HC_time, HC_resourec. chromosome_obj_record={chromosome:[HC_time,HC_record]}
    for o in range(2):
        obj = {m: chroms_obj_record[m][o] for m in front}
        sorted_keys = sorted(obj, key=obj.get)
        distance[sorted_keys[0]] = distance[sorted_keys[len(front) - 1]] = 999999999999
        for i in range(1, len(front) - 1):
            if len(set(obj.values())) == 1:
                distance[sorted_keys[i]] = distance[sorted_keys[i]]
            else:
                distance[sorted_keys[i]] = distance[sorted_keys[i]] + (
                        obj[sorted_keys[i + 1]] - obj[sorted_keys[i - 1]]) / (
                                                   obj[sorted_keys[len(front) - 1]] - obj[sorted_keys[0]])

    return distance

"""Selection - NSGA"""
def selection(population_size, front, chroms_obj_record, total_chromosome):
    N = 0
    new_pop = []
    while N < population_size:
        for i in range(len(front)):
            N = N + len(front[i])
            if N > population_size:
                distance = calculate_crowding_distance(front[i], chroms_obj_record)
                sorted_cdf = sorted(distance, key=distance.get)
                sorted_cdf.reverse()
                for j in sorted_cdf:
                    if len(new_pop) == population_size:
                        break
                    new_pop.append(j)
                break
            else:
                new_pop.extend(front[i])

    population_list = []
    for n in new_pop:
        population_list.append(total_chromosome[n])

    return population_list, new_pop


def fitness_value(chromosome, error_output=False):  # fitness function
    # TODO: our score for valuation is stored in HC_time and HC_resource
    MANDAY = dict()
    HC_time = 0
    HC_resource = 0

    count = 0
    for task in chromosome.chromosome:

        component = task.split(
            '-')
        # take component
        wonum = component[0]
        target_date_begin = component[1]
        end_date_begin = component[2]
        bit_string = component[3]
        # shift = int(bit_date[1])
        shift = int(bit_string[0])
        date_begin = decode_datetime(bit_string[1:-5])
        num_people = bit_string[-5:-3]
        # ===================MODIFY==================
        team_bit = bit_string[-3:]
        team = bit_battery_type[team_bit]
        # ===================MODIFY==================
        # access from dataframe
        est_dur = access_row_by_wonum(wonum)['d_estdur']
        site = access_row_by_wonum(wonum)['site']
        # resouce modify
        # ==================MODIFY===================
        # team = access_row_by_wonum(wonum)['bdpocdiscipline']
        # try:
        #     teamA,teamB = team.split(',')
        # except:
        #     print("Not reconstruction data")
        #     teamA,teamB = team, 'PROC'
        # ==================MODIFY===================
        # convert to datetime type
        try:
            dt_begin = datetime.datetime.strptime(date_begin, '%d/%m/%Y')
            dt_end, shift_end = _cal_end_date(date_begin, shift, est_dur, num_people)
            std_begin = datetime.datetime.strptime(target_date_begin, '%d/%m/%Y')  # start target_day datetime
            etd_end = datetime.datetime.strptime(end_date_begin, '%d/%m/%Y')  # end target_day datetime
            duration_start = (std_begin - dt_begin).days
            duration_end = (dt_end - etd_end).days
            # compute violate point in every element
            if point_duration(duration_start) or point_duration(duration_end):
                if wonum not in chromosome.HC_time:
                    count += 1
                    HC_time += 1
                if error_output:
                    chromosome.HC_time.append(wonum)
                continue

            tup = (team, convert_datetime_to_string(dt_begin), shift, site)

            MANDAY[tup] = MANDAY.get(tup, 0) + 1

            # compute manday resource
            for i in np.arange(0, est_dur, 0.5):
                run_date, run_shift = _cal_end_date(date_begin, shift, i)
                tup_temp = (team, convert_datetime_to_string(run_date), shift, site)
                MANDAY[tup_temp] = MANDAY.get(tup_temp, 0) + 1
        except Exception:
            # invalid days
            if wonum not in chromosome.HC_time:
                HC_time += 1
            if error_output:
                chromosome.HC_time.append(wonum)
    # print(count)
    for key, value in MANDAY.items():
        team, date, shift, site = key
        date = date[:len(date) - 1] + '000' + date[-1]
        data_resource_value = get_resource(team, date, site)
        if data_resource_value == -1 or data_resource_value < value:
            if date not in chromosome.HC_resource:  # gen date with date not in resource
                HC_resource += 1
            if error_output:
                chromosome.HC_resource.append(date)

    return HC_time, HC_resource

# fitness function for caculate score for every chromosome
#
