import copy

from src.NSGA.conditions_nsga import *
from src.all_packages import *
from src.utilities.utils import _cal_end_date

# from src.PMSBX_GA.pmsbx_ga import _generate_parent
# from src.utilities.utils import *
# from .conditions_nsga import *
# from src.utilities.utils import _cal_end_date
# from src.NSGA.conditions_nsga import *
# from ..all_packages import *
data = pd.read_csv('/Users/vinhvu/LiBESS/data/data_test.csv')


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
def _generate_parent(df):
    genes = []
    # for supply_id, start_day, end_date, battery_type in zip(
    #     self.df.supply_id,
    #     self.df.start_day,
    #     self.df.end_date,
    #     self.df.battery_type,
    # ):
    for supply_id, start_day, end_date, battery_type, d_estdur in zip(
            df.supply_id, df.start_day, df.end_date, df.battery_type, df.d_estdur
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


"""Create population function - NSGA"""


class CHROMOSOME_NSGA:
    def __init__(self, df):
        self.HC_resource = []
        self.HC_time = []
        self.df = df
        self.chromosome = self._generate_parent()

    # Generate random date

    def _generate_parent(self):
        genes = []
        # for supply_id, start_day, end_date, battery_type in zip(
        #     self.df.supply_id,
        #     self.df.start_day,
        #     self.df.end_date,
        #     self.df.battery_type,
        # ):
        for supply_id, start_day, end_date, battery_type, d_estdur in zip(
                self.df.supply_id, self.df.start_day, self.df.end_date, self.df.battery_type, self.df.d_estdur
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
    new_population = []
    for i in range(size_of_population):
        new_individual = CHROMOSOME_NSGA(data);
        new_population.append(new_individual)
    new_population = np.asarray(new_population)
    return new_population


"""Select a sub-population for offspring production - NSGA"""


def select_mating_pool(pop, num_parents_mating):
    # shuffling the pop then select top of pops
    pop = np.asarray(pop)
    index = np.random.choice(pop.shape[0], num_parents_mating, replace=False)
    random_individual = pop[index]
    pop = np.delete(pop, index)  # split current pop into remain_pop and mating_pool
    return random_individual


"""Crossover - NSGA"""


# def crossover(parents):
#     mating_pool = copy.deepcopy(parents)
#     offsprings = []
#     while mating_pool.size > 0:
#         mating_idx = np.random.choice(mating_pool.shape[0], 2, replace=False)
#         mating_parents = mating_pool[mating_idx]
#         parent_1 = mating_parents[0]
#         parent_2 = mating_parents[1]
#         swap_task_pos = random.randrange(parent_1.CHROMOSOME_NSGA.shape[0])
#         # print("parent_1: ", parent_1[swap_task_pos])
#         # Tính toán phạm vi cho hàm random.sample()
#         # start_index = parent_1[0].rfind("-") + 1  # Vị trí của ký tự '-' đầu tiên
#         # end_index = len(parent_1[0]) - 1  # Độ dài của chuỗi - 1
#
#         # # Tạo danh sách các index không thuộc vào khoảng [13:16] (không đổi loại pin)
#         # valid_indexes = list(range(start_index, start_index + 12)) + list(
#         #     range(start_index + 16, end_index + 1)
#         # )
#
#         # # Chọn ngẫu nhiên 2 điểm cắt từ danh sách các index hợp lệ
#         # crossover_point = random.sample(valid_indexes, 2)
#         crossover_point = random.sample(
#             range(parent_1[0].rfind("-") + 1, len(parent_1[0])),
#             2,
#         )
#         crossover_point.sort()
#         # print(crossover_point)
#         offspring_1 = (
#             parent_1[swap_task_pos][0 : crossover_point[0]]
#             + parent_2[swap_task_pos][crossover_point[0] : crossover_point[1]]
#             + parent_1[swap_task_pos][crossover_point[1] :]
#         )
#         offspring_2 = (
#             parent_2[swap_task_pos][0 : crossover_point[0]]
#             + parent_1[swap_task_pos][crossover_point[0] : crossover_point[1]]
#             + parent_2[swap_task_pos][crossover_point[1] :]
#         )
#         parent_1[swap_task_pos] = offspring_1
#         parent_2[swap_task_pos] = offspring_2
#         offsprings.append(parent_1)
#         offsprings.append(parent_2)
#         mating_pool = np.delete(mating_pool, list(mating_idx), axis=0)
#
#     return np.array(offsprings)
def find_point(test_str):
    begin = test_str.rfind('-') + 1
    shift = test_str[begin]
    # rand_date = test_str[begin + 1: begin + 12]
    # num_people = test_str[begin + 12 : begin + 14]
    # team_gen = test_str[begin + 14 : begin + 17]

    crossover_rand_date = random.sample(range(begin + 1, begin + 12), 2)
    # crossover_num_people = random.sample(range(begin + 12,begin + 14),2)
    crossover_rand_date.sort()

    return begin, crossover_rand_date


def _str_time_prop(start, end, time_format, prop):
    stime = datetime.datetime.strptime(start, time_format)
    etime = datetime.datetime.strptime(end, time_format)
    ptime = stime + prop * (etime - stime)
    ptime = ptime.strftime("%d/%m/%Y")
    return ptime


def random_date(start, end, prop):  # 0001 = current year, 0002 = next year
    # generate date in current data
    sched_start = _str_time_prop(start, end, "%d/%m/%Y", prop)
    date_sched_start = format(int(sched_start[:2]), '05b')
    month_sched_start = format(int(sched_start[3:5]), '04b')
    year_sched_start = format(int(sched_start[6:]), '02b')
    sched_start = ''.join([date_sched_start, month_sched_start, year_sched_start])
    return sched_start


def crossover(parents):
    mating_pool = copy.deepcopy(parents)
    offsprings = []
    while mating_pool.size > 0:
        mating_idx = np.random.choice(mating_pool.shape[0], 2, replace=True)
        mating_parents = mating_pool[mating_idx]
        parent_1 = mating_parents[0]
        parent_2 = mating_parents[1]
        swap_task_pos = random.randrange(parent_1.chromosome.shape[0])
        # print('parent_1: ',parent_1.chromosome[swap_task_pos])
        begin, crossover_rand_date = find_point(parent_1.chromosome[0])

        parent_1_str = str(parent_1.chromosome[swap_task_pos])
        parent_2_str = str(parent_2.chromosome[swap_task_pos])
        component_parent_1 = parent_1_str.split('-')
        component_parent_2 = parent_2_str.split('-')
        # take component
        target_date_begin_1 = component_parent_1[1]
        end_date_begin_1 = component_parent_1[2]
        target_date_begin_2 = component_parent_2[1]
        end_date_begin_2 = component_parent_2[2]
        i = 0
        while True:
            # hyperparameter track cross_over_shift
            offspring_1 = parent_1_str[0:begin]
            offspring_2 = parent_2_str[0:begin]
            if random.random() > 0.5:
                offspring_1 += parent_2_str[begin]
                offspring_2 += parent_1_str[begin]
            else:
                offspring_1 += parent_1_str[begin]
                offspring_2 += parent_2_str[begin]
            # hyperparameter track cross_over_date
            if random.random() > 0.3:
                # begin_bit_date, end_bit_date = crossover_rand_date
                # offspring_1 += parent_1_str[begin + 1: begin_bit_date] + parent_2_str[
                #                                                          begin_bit_date:end_bit_date + 1] + parent_1_str[
                #                                                                                             end_bit_date + 1:begin + 12]
                # offspring_2 += parent_2_str[begin + 1: begin_bit_date] + parent_1_str[
                #                                                          begin_bit_date:end_bit_date + 1] + parent_2_str[
                #                                                                                             end_bit_date + 1:begin + 12]
                # _________________________ regenerate date ______________________________________
                offspring_1 += random_date(target_date_begin_1, end_date_begin_1, random.random())
                offspring_2 += random_date(target_date_begin_2, end_date_begin_2, random.random())

            else:
                offspring_1 += parent_1_str[begin + 1:begin + 12]
                offspring_2 += parent_2_str[begin + 1:begin + 12]
            # hyperparameter track num_people
            if random.random() > 0.7:
                offspring_1 += parent_2_str[begin + 12:begin + 14]
                offspring_2 += parent_1_str[begin + 12:begin + 14]
            else:
                offspring_1 += parent_1_str[begin + 12:begin + 14]
                offspring_2 += parent_2_str[begin + 12:begin + 14]

            # team keep stable
            offspring_1 += parent_1_str[begin + 14:]
            offspring_2 += parent_2_str[begin + 14:]
            i += 1
            if i > 0:
                parent_1.chromosome[swap_task_pos] = np.str_(offspring_1)
                parent_2.chromosome[swap_task_pos] = np.str_(offspring_2)
                break
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
        mutate_flag = 0
        for index, task in enumerate(chromosome.chromosome):
            if index >= len(chromosome.chromosome) - 3:
                continue
            rate = random.uniform(0, 1)
            if rate < random_rate:
                index = random.randrange(task.rfind('-') + 1, len(task) - 1)
                newGene, alternate = random.sample(geneSet, 2)
                mutate_gene = alternate if newGene == task[index] else newGene
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

    for o in range(2):  # Assuming there are two objectives.
        # Create a dictionary with members of the front as keys and their objective values as items.
        obj = {}
        for m in front:
            if isinstance(chroms_obj_record[m][o], list):
                obj[m] = tuple(chroms_obj_record[m][o])
            else:
                # Handle the situation when it's not a list.
                obj[m] = (chroms_obj_record[m][o],)
        # Sort the keys of the dictionary based on their corresponding objective value.
        sorted_keys = sorted(obj, key=obj.get)

        # Assign a large distance to the boundary points.
        distance[sorted_keys[0]] = distance[sorted_keys[len(front) - 1]] = 999999999999

        for i in range(1, len(front) - 1):
            # If all values are identical, assign a zero distance.
            if len(set(obj.values())) == 1:
                distance[sorted_keys[i]] = 0
            else:
                # Calculate crowding distance based on the objective values.
                # Note: We need to ensure that the objective values are numeric and subtractable.
                prev_val = obj[sorted_keys[i - 1]]
                next_val = obj[sorted_keys[i + 1]]
                # Calculate the difference between the next and previous values.
                differences = [n - p for n, p in zip(next_val, prev_val)]
                # Calculate the normalized crowding distance.
                norm = max(obj[sorted_keys[-1]]) - min(obj[sorted_keys[0]])
                if norm == 0:
                    distance[sorted_keys[i]] += 0
                else:
                    crowding_distance = sum(abs(d) for d in differences) / norm
                    distance[sorted_keys[i]] += crowding_distance

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
    for gen in chromosome.chromosome:

        component = gen.split(
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
        team = battery_type_bit_convert[team_bit]
        # ===================MODIFY==================
        # access from dataframe
        est_dur = access_row_by_wonum(wonum)['d_estdur']
        site = access_row_by_wonum(wonum)['site']  # edir
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
            if point_duration(duration_start) or point_duration(duration_end):  # D
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
# def cal_hc_values(chromosome):
#     HC_TIME_total = 0
#     HC_RESOURCE_total = 0
#
#     # for gen in chromosome.chromosome:
#         # Tính toán vi phạm nguồn lực từ hard_constraint_1
#     HC_RESOURCE_count = hard_constraint_1(chromosome)
#     HC_RESOURCE_total += HC_RESOURCE_count
#
#     # Tính toán vi phạm thời gian từ hard_constraint_2 và hard_constraint_3
#     HC_TIME_count = hard_constraint_2(chromosome) + hard_constraint_3(chromosome)
#     HC_TIME_total += HC_TIME_count
#
#     return HC_TIME_total, HC_RESOURCE_total


def cal_hc_time_and_resource(chromosome):
    HC_TIME = []
    HC_RESOURCE = []
    result, HC_resource_count = manday_chromosome(chromosome)
    # Giả sử deadline_count được tính toán trong manday_chromosome và được trả về như một phần của result
    # Điều này yêu cầu bạn điều chỉnh manday_chromosome để trả về cả HC_resource_count và deadline_count
    deadline_count = (result - HC_resource_count) / 5  # Tính lại từ công thức result = HC_count + deadline_count * 5
    HC_TIME.append(deadline_count)
    HC_RESOURCE.append(HC_resource_count)
    return deadline_count, HC_resource_count
