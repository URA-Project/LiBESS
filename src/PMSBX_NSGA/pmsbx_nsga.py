import copy
import math
import sys
from all_packages import *
from utilities.utils import *
from collections import defaultdict
sys.path.insert(0, 'C:/Users/trong.le-van/OneDrive - Ban Vien Corporation/HCMUT_OneDrive/URA/Paper/LiBESS-NSGA_II/LiBESS-NSGA-II-V2/src')
data = pd.read_csv('C:/Users/trong.le-van/OneDrive - Ban Vien Corporation/HCMUT_OneDrive/URA/Paper/LiBESS-NSGA_II/LiBESS-NSGA-II-V2/data/data_test.csv')

"""Create population function - NSGA"""

class CHROMOSOME_PMSBX_NSGA:
    def __init__(self, df):
        self.HC_resource = []
        self.HC_time = []
        self.df = df
        self.chromosome = self._generate_parent()

    # Generate random date
    def _generate_parent(self):
        genes = []
        for supply_id, start_day, end_date, battery_type, d_estdur in zip(
            self.df.supply_id, self.df.start_day, self.df.end_date, self.df.battery_type, self.df.d_estdur
        ):
            rand_date = random_datetime(start_day, end_date)
            routine = random.choice([0, 1])
            battery_type_list = battery_type.split("|")
            battery_type_gen = random.choice(battery_type_list)
            battery_type_gen = battery_type_dec[battery_type_gen]

            num_battery = random.randint(2, 10)
            decstring = "-".join(
                [rand_date, str(routine), str(battery_type_gen), str(num_battery)]
            )
            chromosome = "-".join([supply_id, start_day, end_date, decstring])
            genes.append(chromosome)
        return np.asarray(genes)

def init_population(size_of_population):
    new_population = []
    for i in range(size_of_population):
        new_individual = CHROMOSOME_PMSBX_NSGA(data)
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

def find_point(test_str):
    begin = test_str.rfind('-') + 1
    shift = test_str[begin]

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

def crossover(mating_pool, distribution_index):
    offspring = []
    parents = copy.deepcopy(mating_pool)

    for k in range(0, parents.shape[0], 2):
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
        offspring_1 = []
        offspring_2 = []
        for task1, task2 in zip(chromosome_1, chromosome_2):
            gen_1 = parser_gen_pmsbx(task1)
            gen_2 = parser_gen_pmsbx(task2)
            new_gen_1, new_gen_2 = crossover_calculation(
                gen_1, gen_2, distribution_index
            )
            new_gen_1_string = convert_to_string_format(new_gen_1, gen_1)
            new_gen_2_string = convert_to_string_format(new_gen_2, gen_2)
            offspring_1.append(new_gen_1_string)
            offspring_2.append(new_gen_2_string)

        temp_1 = parents[parent1_idx]
        temp_1.chromosome = offspring_1
        temp_2 = parents[parent1_idx]
        temp_2.chromosome = offspring_2
        offspring.append(temp_1)
        offspring.append(temp_2)
    return np.array(offspring)

def convert_to_string_format(vector, gene):
    # gene(id, start_date, end_date, scheduled_date, routine, battery_type, num_battery)
    # v = (routine,(scheduled_date − start_date), battery_type, num_battery)
    start_date = datetime.datetime.strptime(gene.start_date, date_format)
    scheduled_date = datetime.timedelta(days=vector.difference_date) + start_date
    scheduled_date_string = scheduled_date.strftime(date_format)

    decstring = "-".join(
        [
            scheduled_date_string,
            str(vector.routine),
            str(vector.battery_type),
            str(vector.num_battery),
        ]
    )
    new_gene = "-".join([gene.id, gene.start_date, gene.end_date, decstring])
    return new_gene

def crossover_calculation(gen_1, gen_2, distribution_index):
    # Khoi tao bien group selected variables
    # Gen_structure = (id, start_date, end_date, scheduled_date, routine, battery_type, num_battery)
    # v = (routine,(scheduled_date − start_date), battery_type, num_battery)
    beta_para = 0

    diff_date_gen_1 = difference_date(gen_1.scheduled_date, gen_1.start_date)
    diff_date_gen_2 = difference_date(gen_1.scheduled_date, gen_1.start_date)

    candidate_1 = (0, 0, 0, 0)
    candidate_2 = (0, 0, 0, 0)

    check = False
    while check == False:
        random_rate = generate_random_number()
        if random_rate <= 0.5:
            beta_para = power_of_fraction(2 * random_rate, 1, distribution_index + 1)
        elif random_rate > 0.5:
            beta_para = power_of_fraction(
                (2 - 2 * random_rate), 1, distribution_index + 1
            )

        vector_v1 = (
            int(gen_1.routine),
            diff_date_gen_1.days,
            int(gen_1.battery_type),
            int(gen_1.num_battery),
        )
        vector_v2 = (
            int(gen_2.routine),
            diff_date_gen_2.days,
            int(gen_2.battery_type),
            int(gen_2.num_battery),
        )

        # v1_new = 0.5 × [(1 + β)υ1 + (1 − β)υ2]
        candidate_1 = scalar_multiply_v1_crossover(beta_para, vector_v1, vector_v2)
        # v2_new = 0.5 × [(1 - β)υ1 + (1 + β)υ2]
        candidate_2 = scalar_multiply_v2_crossover(beta_para, vector_v1, vector_v2)

        # Check for violations of each variable in v_1 and v_2
        check = check_violations(candidate_1, candidate_2)
    return candidate_1, candidate_2

def check_violations(candidate_1, candidate_2):
    # ['routine', 'difference_date', 'battery_type', 'num_battery']

    # check  routine ∈ {0, 1} is the ROUTINE variable.
    if candidate_1.routine <= 0 and candidate_1.routine >= 1:
        return False
    if candidate_2.routine <= 0 and candidate_2.routine >= 1:
        return False

    # difference_date ∈ [0..30] is the difference between SCHEDULED DATE and the START DATE.
    if candidate_1.difference_date <= 0 and candidate_1.difference_date >= 30:
        return False
    if candidate_2.difference_date <= 0 and candidate_2.difference_date >= 30:
        return False

    # battery_type ∈ [1..5] is the BATTERY TYPE variable
    if candidate_1.battery_type <= 1 and candidate_1.battery_type >= 5:
        return False
    if candidate_2.battery_type <= 1 and candidate_2.battery_type >= 5:
        return False

    # num_battery ∈ [1..10] is the NUMBER OF BATTERIES variable
    if candidate_1.num_battery <= 1 and candidate_1.num_battery >= 10:
        return False
    if candidate_2.num_battery <= 1 and candidate_2.num_battery >= 10:
        return False
    return True

"""Mutation - NSGA"""

def mutation(offspring_crossover, distribution_index):
    new_offspring = []
    offspring = copy.deepcopy(offspring_crossover)
    for index in range(len(offspring)):
        chromosome = offspring[index].chromosome
        offspring_temp = []
        for index_gene in range(len(chromosome)):
            gene = parser_gen_pmsbx(chromosome[index_gene])
            vector = mutation_calculation(gene, distribution_index)
            new_gene_string = convert_to_string_format(vector, gene)
            offspring_temp.append(new_gene_string)
        temp = offspring[index]
        temp.chromosome = offspring_temp
        new_offspring.append(temp)
    return np.array(new_offspring)

def mutation_calculation(gene, distribution_index):
    # Khoi tao bien group selected variables
    # v = (routine,(scheduled_date − start_date), battery_type, num_battery)
    diff_date_gen_1 = difference_date(gene.scheduled_date, gene.start_date)
    vector = (
        int(gene.routine),
        diff_date_gen_1.days,
        int(gene.battery_type),
        int(gene.num_battery),
    )
    delta_para = 0
    new_gen = (0, 0, 0, 0)
    # Generate a random number ε ∈ R, where 0 ≤ ε ≤ 1
    random_rate = generate_random_number()
    if random_rate <= 0.5:
        delta_para = power_of_fraction(2 * random_rate, 1, distribution_index + 1) - 1
    else:
        delta_para = 1 - power_of_fraction(
            (2 - 2 * random_rate), 1, distribution_index + 1
        )
    # temp = Vector(*parents)
    # vector = (temp.routine, temp.difference_date, temp.battery_type, temp.num_battery)
    vector = np.array([int(gene.routine), diff_date_gen_1.days, int(gene.battery_type), int(gene.num_battery)])
    new_gen = scalar_multiply_motation(vector, delta_para, random_rate)
    return new_gen

def scalar_multiply_motation(vector, delta, epsilon):
    new_vector = (0, 0, 0, 0)
    # ['routine', 'difference_date', 'battery_type', 'num_battery']
    random_routine = random.randint(0, 1)
    random_diff_date = random.randint(0, 30)
    random_battery_type = random.randint(1, 5)
    random_num_battery = random.randint(1, 10)
    vector_random = (
        random_routine,
        random_diff_date,
        random_battery_type,
        random_num_battery,
    )
    if epsilon <= 0.5:
        # v_new = v + δ × [υ − (a1, a2, a3, a4)] for ε ≤ 0.5
        new_vector = tuple(
            u1 + delta * (u1 - u2) for u1, u2 in zip(vector, vector_random)
        )
    else:
        # v_new = v + δ × [(a1, a2, a3, a4) - υ] for ε ≤ 0.5
        new_vector = tuple(
            u1 + delta * (u2 - u1) for u1, u2 in zip(vector, vector_random)
        )
    rounded_vector = tuple(int(math.ceil(element)) for element in new_vector)
    return Vector(*rounded_vector)

'''-------non-dominated sorting function-------'''
def non_dominated_sorting(population_size, chroms_obj_record):
    s, n = {}, {}
    front, rank = {}, {}
    front[0] = []
    for p in range(population_size * 2):
        s[p] = []
        n[p] = 0
        for q in range(population_size * 2):
            #TODO: define dominate operator in the if statement
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

    # Lấy các điểm từ chroms_obj_record
    pareto_points = []
    for idx_list in front.values():
        for idx in idx_list:
            pareto_points.append(chroms_obj_record[idx])

    points = {}
    for idx, point in enumerate(pareto_points):
        points[idx] = point
    return front, chroms_obj_record

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
                # This is just an example, you need to replace it with your own handling code.
                # print(f"chroms_obj_record[{m}][{o}] is not a list.")
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

def cal_hc_time_and_resource(chromosome):
    HC_TIME = []
    HC_RESOURCE = []
    result, HC_resource_count, total_capacity = manday_chromosome(chromosome)
    # Giả sử deadline_count được tính toán trong manday_chromosome và được trả về như một phần của result
    # Điều này yêu cầu bạn điều chỉnh manday_chromosome để trả về cả HC_resource_count và deadline_count
    deadline_count = (result - HC_resource_count) / 5  # Tính lại từ công thức result = HC_count + deadline_count * 5
    HC_TIME.append(total_capacity)
    HC_RESOURCE.append(HC_resource_count)
    return total_capacity, HC_resource_count

def manday_chromosome(total_chromosome):
    chromosome = copy.deepcopy(total_chromosome)
    HC_count = 0
    deadline_count = 0
    result = 0
    total_capacity = 0
    BATDAY = defaultdict(float)

    for gen in chromosome.chromosome:
        gen_parser = parser_gen_pmsbx(gen)
        d_estdur = access_row_by_wonum(gen_parser.id)["d_estdur"]
        device = access_row_by_wonum(gen_parser.id)["device_type"]
        dealine = access_row_by_wonum(gen_parser.id)["end_date"]
        date_dealine = datetime.datetime.strptime(dealine, "%d/%m/%Y")
        date_begin = datetime.datetime.strptime(gen_parser.scheduled_date, "%d/%m/%Y")

        num_date = round(d_estdur)
        check_dealine = date_begin + datetime.timedelta(days=num_date)
        if date_dealine < check_dealine:
            deadline_count += 1
        for i in range(int(d_estdur * 2)):  # Loop over half days
            num_date = i / 2
            run_date = date_begin + datetime.timedelta(days=num_date)
            tup_temp = (
                battery_type_dec_convert[gen_parser.battery_type],
                run_date.strftime("%d/%m/%Y"),
                device,
            )
            BATDAY[tup_temp] += 0.5

        num_batteries = gen_parser.num_battery
        date = validate_date_format(gen_parser.scheduled_date)
        if date != -1:
            data_value = get_resource(battery_type_dec_convert[gen_parser.battery_type], date, device)
            total_capacity = int(num_batteries) * int(data_value) + total_capacity
        else: 
            deadline_count += 1

    for key, value in BATDAY.items():
        bat_type, date, device = key
        data_resource_value = get_resource(bat_type, date, device)
        value_KAh = value * 5

        if data_resource_value == -1 or data_resource_value < value_KAh:
            HC_count += 1

    result =  HC_count + deadline_count*5

    return result, HC_count, total_capacity

def validate_date_format(date_str):
    try:
        date_obj = datetime.datetime.strptime(date_str, "%d/%m/%Y")
        return date_obj.strftime("%d/%m/%Y")
    except ValueError:
        return -1