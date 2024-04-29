import copy
import sys

sys.path.insert(
    0,
    "/Users/mac/Library/CloudStorage/OneDrive-Personal/Study/URA/GA_Emerging_Papers/LiBESS-MAIN/src",
)
from all_packages import *
from utilities.utils import *
from collections import defaultdict

data = pd.read_csv(
    "/Users/mac/Library/CloudStorage/OneDrive-Personal/Study/URA/GA_Emerging_Papers/LiBESS-MAIN/data/data_test.csv"
)

"""Create population function - NSGA"""


class CHROMOSOME_GA:
    def __init__(self, df):
        self.HC_resource = []
        self.HC_time = []
        self.df = df
        self.chromosome = self._generate_parent()

    # Generate random date
    def _generate_parent(self):
        chromosome = []
        for supply_id, start_day, end_date, battery_type, d_estdur in zip(
            self.df.supply_id,
            self.df.start_day,
            self.df.end_date,
            self.df.battery_type,
            self.df.d_estdur,
        ):
            rand_date = random_date_bit(start_day, end_date, random.random())
            routine = random.choice([0, 1])
            battery_type = battery_type.split("|")
            battery_type_gen = random.choice(battery_type)
            battery_type_gen = battery_type_bit[battery_type_gen]
            random_num_battery = random.randint(0, 10)
            num_battery = format(random_num_battery, "04b")
            bitstring = "".join(
                [rand_date, str(routine), battery_type_gen, num_battery]
            )
            genes = "-".join([supply_id, start_day, end_date, bitstring])
            chromosome.append(genes)
        return np.asarray(chromosome)


def init_population(size_of_population):
    new_population = []
    for i in range(size_of_population):
        new_individual = CHROMOSOME_GA(data)
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
    begin = test_str.rfind("-") + 1
    shift = test_str[begin]
    crossover_rand_date = random.sample(range(begin + 1, begin + 12), 2)
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
    date_sched_start = format(int(sched_start[:2]), "05b")
    month_sched_start = format(int(sched_start[3:5]), "04b")
    year_sched_start = format(int(sched_start[6:]), "02b")
    sched_start = "".join([date_sched_start, month_sched_start, year_sched_start])
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
        begin, crossover_rand_date = find_point(parent_1.chromosome[0])

        parent_1_str = str(parent_1.chromosome[swap_task_pos])
        parent_2_str = str(parent_2.chromosome[swap_task_pos])
        component_parent_1 = parent_1_str.split("-")
        component_parent_2 = parent_2_str.split("-")
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
                offspring_1 += random_date(
                    target_date_begin_1, end_date_begin_1, random.random()
                )
                offspring_2 += random_date(
                    target_date_begin_2, end_date_begin_2, random.random()
                )

            else:
                offspring_1 += parent_1_str[begin + 1 : begin + 12]
                offspring_2 += parent_2_str[begin + 1 : begin + 12]
            # hyperparameter track num_people
            if random.random() > 0.7:
                offspring_1 += parent_2_str[begin + 12 : begin + 14]
                offspring_2 += parent_1_str[begin + 12 : begin + 14]
            else:
                offspring_1 += parent_1_str[begin + 12 : begin + 14]
                offspring_2 += parent_2_str[begin + 12 : begin + 14]

            # team keep stable
            offspring_1 += parent_1_str[begin + 14 :]
            offspring_2 += parent_2_str[begin + 14 :]
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
    geneSet = ["0", "1"]
    pop = copy.deepcopy(population)

    mutation_offspring = []
    # Mutation changes a number of genes as defined by the num_mutations argument. The changes are random.
    for chromosome in pop:
        new_chromosome = []
        for index, task in enumerate(chromosome.chromosome):
            rate = random.uniform(0, 1)
            if rate < random_rate:
                index = random.randrange(task.rfind("-") + 1, len(task) - 1)
                newGene, alternate = random.sample(geneSet, 2)
                mutate_gene = alternate if newGene == task[index] else newGene
                task = task[:index] + mutate_gene + task[index + 1 :]
            new_chromosome.append(task)
        chromosome.chromosome = new_chromosome
        chromosome.HC_time = []
        chromosome.HC_resource = []
        mutation_offspring.append(chromosome)
    return np.asarray(mutation_offspring)


"""Selection - NSGA"""


def selection(popsize, chroms_obj_record, total_chromosome):
    # Sắp xếp các cặp (index, value) dựa trên giá trị của value
    sorted_pairs = sorted(chroms_obj_record.items(), key=lambda x: x[1])
    # Lấy chỉ mục từ các cặp đã sắp xếp
    sorted_indices = [pair[0] for pair in sorted_pairs]
    new_population = []
    new_chroms_obj_record = {}
    count_obj_record = 0

    i = 0
    for index_value in sorted_indices:
        if (i + 1) <= popsize:
            if chroms_obj_record[index_value] not in new_chroms_obj_record.values():
                new_chroms_obj_record[count_obj_record] = chroms_obj_record[index_value]
                temp_chromosome = total_chromosome[index_value]
                chromosome = total_chromosome[index_value].chromosome
                temp_chromosome.chromosome = np.asarray(chromosome)
                # temp_chromosome.chromosome = np.asarray(chromosome)
                temp_chromosome.HC_time = []
                temp_chromosome.HC_resource = []
                new_population.append(temp_chromosome)
                count_obj_record += 1
                i = i + 1

    return np.asarray(new_population), new_chroms_obj_record


def cal_hc_time_and_resource(chromosome):
    HC_TIME = []
    HC_RESOURCE = []
    resource_and_deadline, total_capacity = manday_chromosome(chromosome)
    # Giả sử deadline_count được tính toán trong manday_chromosome và được trả về như một phần của result
    # Điều này yêu cầu bạn điều chỉnh manday_chromosome để trả về cả HC_resource_count và deadline_count
    HC_TIME.append(total_capacity)
    HC_RESOURCE.append(resource_and_deadline)
    return total_capacity, resource_and_deadline


def manday_chromosome(chromosome):
    HC_count = 0
    deadline_count = 0
    result = 0
    total_capacity = 0
    BATDAY = defaultdict(float)

    for gen in chromosome.chromosome:
        gen_parser = parser_gen_ga(gen)
        d_estdur = access_row_by_wonum(gen_parser.id)["d_estdur"]
        device = access_row_by_wonum(gen_parser.id)["device_type"]

        # Sanitize and parse scheduled_date
        sanitized_scheduled_date = sanitize_date(gen_parser.scheduled_date)
        sanitized_scheduled_date = validate_date_format(sanitized_scheduled_date)
        if sanitized_scheduled_date != -1:
            date_begin = datetime.datetime.strptime(
                sanitized_scheduled_date, "%d/%m/%Y"
            )
        else:
            deadline_count += 1
            continue

        deadline = access_row_by_wonum(gen_parser.id)["end_date"]
        date_deadline = datetime.datetime.strptime(deadline, "%d/%m/%Y")

        num_date = round(d_estdur)
        check_deadline = date_begin + datetime.timedelta(days=num_date)
        if date_deadline < check_deadline:
            deadline_count += 1
        for i in range(int(d_estdur * 2)):  # Loop over half days
            num_date = i / 2
            run_date = date_begin + datetime.timedelta(days=num_date)
            tup_temp = (gen_parser.battery_type, run_date.strftime("%d/%m/%Y"), device)
            BATDAY[tup_temp] += 0.5

        num_batteries = gen_parser.num_battery
        date = validate_date_format(gen_parser.scheduled_date)
        if date != -1:
            data_value = get_resource(gen_parser.battery_type, date, device)
            total_capacity = num_batteries * int(data_value) + total_capacity
        else:
            deadline_count += 1

    for key, value in BATDAY.items():
        bat_type, date, device = key
        data_resource_value = get_resource(bat_type, date, device)
        value_KAh = value * 5
        if data_resource_value == -1 or data_resource_value < value_KAh:
            HC_count += 1

    result = HC_count + deadline_count * 5
    return result, HC_count, total_capacity


def sanitize_date(date_str):
    parts = date_str.split("/")
    # Correct the day part if it's '0'.
    if parts[0] == "0":
        parts[0] = "01"
    corrected_date_str = "/".join(parts)
    return corrected_date_str


def validate_date_format(date_str):
    try:
        date_obj = datetime.datetime.strptime(date_str, "%d/%m/%Y")
        return date_obj.strftime("%d/%m/%Y")
    except ValueError:
        return -1


def save_output(chromosome, index):
    total_battery = 0
    total_capacity = 0
    BATDAY = dict()
    resources = 0
    deadline_count = 0
    BATDAY = defaultdict(float)

    for gen in chromosome.chromosome:
        gen_parser = parser_gen_ga(gen)
        d_estdur = access_row_by_wonum(gen_parser.id)["d_estdur"]
        device = access_row_by_wonum(gen_parser.id)["device_type"]
        dealine = access_row_by_wonum(gen_parser.id)["end_date"]
        date_dealine = datetime.datetime.strptime(dealine, "%d/%m/%Y")

        # Sanitize and parse scheduled_date
        sanitized_scheduled_date = sanitize_date(gen_parser.scheduled_date)
        sanitized_scheduled_date = validate_date_format(sanitized_scheduled_date)
        if sanitized_scheduled_date != -1:
            date_begin = datetime.datetime.strptime(
                sanitized_scheduled_date, "%d/%m/%Y"
            )
        else:
            deadline_count += 1
            continue

        num_date = round(d_estdur)
        check_dealine = date_begin + datetime.timedelta(days=num_date)
        check_dealine_string = check_dealine.strftime("%d/%m/%Y")
        check_dealine_string = validate_date_format(check_dealine_string)
        if check_dealine == -1:
            deadline_count += 1
            continue
        check_dealine = datetime.datetime.strptime(check_dealine_string, "%d/%m/%Y")
        if date_dealine < check_dealine:
            deadline_count += 1
        for i in range(int(d_estdur * 2)):  # Loop over half days
            num_date = i / 2
            run_date = date_begin + datetime.timedelta(days=num_date)
            tup_temp = (
                gen_parser.battery_type,
                run_date.strftime("%d/%m/%Y"),
                device,
            )
            BATDAY[tup_temp] += 0.5

    for key, value in BATDAY.items():
        bat_type, date, device = key
        data_resource_value = get_resource(bat_type, date, device)
        # data_resource_value = data_resource_value * 1.25
        value_KAh = value * 5

        if data_resource_value == -1 or data_resource_value < value_KAh:
            resources += 1

    for gen in chromosome.chromosome:
        # gen_parser = "id","start_date","end_date","scheduled_date","routine","battery_type","num_battery"
        gen_parser = parser_gen_ga(gen)
        total_battery = total_battery + int(gen_parser.num_battery)

        sanitized_scheduled_date = sanitize_date(gen_parser.scheduled_date)
        sanitized_scheduled_date = validate_date_format(sanitized_scheduled_date)
        if sanitized_scheduled_date != -1:
            ngay_thang_dt = datetime.datetime.strptime(
                sanitized_scheduled_date, "%d/%m/%Y"
            )
        else:
            continue
        scheduled_date = ngay_thang_dt.strftime("%d/%m/%Y")
        type_battery = gen_parser.battery_type
        num_batteries = int(gen_parser.num_battery)
        device = access_row_by_wonum(gen_parser.id)["device_type"]
        data_resource_value = get_resource(type_battery, scheduled_date, device)
        total_capacity = num_batteries * int(data_resource_value) + total_capacity

    # Writing results to file
    if ((index + 1) % 10 == 0) or (index == 0):
        output_file = "/Users/mac/Library/CloudStorage/OneDrive-Personal/Study/URA/GA_Emerging_Papers/LiBESS-MAIN/output/output_ga.txt"
        with open(output_file, "a") as file:
            file.write("++++++++++Lan lap thu  {} ++++++++ \n".format(index + 1))
            file.write("Dealine {}:  {}\n".format(index + 1, deadline_count))
            file.write("Total battery {}:  {}\n".format(index + 1, total_battery))
            file.write("Total capacity {}:  {}\n".format(index + 1, total_capacity))
            file.write("Resources {}:  {}\n".format(index + 1, resources))
