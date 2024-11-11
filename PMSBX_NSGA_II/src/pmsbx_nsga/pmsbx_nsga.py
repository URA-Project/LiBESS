import copy
import random
from datetime import timedelta
from typing import Dict, List, Tuple
import numpy as np
from src.parameters import CrossoverParams, MutationParams
from src.pmsbx_nsga.init import (
    Chromosome,
    Gene,
    Individual,
    Population,
)

from src.pmsbx_nsga.utils import Utils


class GeneticOperators:

    def __init__(self, supply_orders=None, session_capacity_threshold=None, population_size=0):
        if (
            supply_orders is not None
            and not supply_orders.empty
            and population_size > 0
        ):
            self.population = Population(supply_orders, session_capacity_threshold, population_size)
        else:
            self.population = []

    def select_mating_pool(self, population: Population, num_parents_mating: int = 0):
        mating_pool = Population()
        individuals = copy.deepcopy(population.individuals)
        for individual in individuals[:num_parents_mating]:
            mating_pool.individuals.append(individual)
        return mating_pool

    def crossover(
        self,
        mating_pool: Population,
        crossover_params: CrossoverParams,
        power_supply_capacity_max: int,
    ):
        offspring = Population()
        parents = copy.deepcopy(mating_pool)

        while len(parents.individuals) > 0:
            # Randomly select two distinct individuals from the population
            mating_idx = np.random.choice(len(parents.individuals), 2, replace=False)
            # Select the parents based on the indices
            mating_parents = [
                parents.individuals[mating_idx[0]],
                parents.individuals[mating_idx[1]],
            ]

            # Ensure mating_parents[0] and mating_parents[1] are instances of Individual
            individual1: Individual = mating_parents[0]
            individual2: Individual = mating_parents[1]

            # Access chromosomes in individual1 and individual2
            chromosomes_of_individual1 = individual1.chromosomes
            chromosomes_of_individual2 = individual2.chromosomes
            new_individual_1 = Individual()
            new_individual_2 = Individual()
            for chromosome_1, chromosome_2 in zip(
                chromosomes_of_individual1, chromosomes_of_individual2
            ):
                chromosome_1: Chromosome
                chromosome_2: Chromosome
                new_chromosome_1 = Chromosome(
                    total_expected=chromosome_1.total_expected,
                    device_type=chromosome_1.device_type,
                    battery_type_list=chromosome_1.battery_type_list,
                )
                new_chromosome_2 = Chromosome(
                    total_expected=chromosome_2.total_expected,
                    device_type=chromosome_2.device_type,
                    battery_type_list=chromosome_2.battery_type_list,
                )
                for gen_1, gen_2 in zip(chromosome_1.genes, chromosome_2.genes):
                    new_gen_1, new_gen_2 = self._crossover_calculation(
                        gen_1,
                        gen_2,
                        crossover_params,
                        power_supply_capacity_max,
                    )
                    new_chromosome_1.genes.append(new_gen_1)
                    new_chromosome_2.genes.append(new_gen_2)
                new_individual_1.chromosomes.append(new_chromosome_1)
                new_individual_2.chromosomes.append(new_chromosome_2)
            offspring.individuals.append(new_individual_1)
            offspring.individuals.append(new_individual_2)
            parents.individuals = np.delete(
                parents.individuals, list(mating_idx), axis=0
            )
        return offspring

    def mutation(
        self,
        offspring_cros: Population,
        mutation_params: MutationParams,
        power_supply_capacity_max: int,
    ):
        offspring_crossover = copy.deepcopy(offspring_cros)
        new_offspring = Population()
        for individual in offspring_crossover.individuals:
            mutated_individual = Individual()
            for chromosome in individual.chromosomes:
                chromosome: Chromosome
                new_chromosome = Chromosome(
                    total_expected=chromosome.total_expected,
                    device_type=chromosome.device_type,
                    battery_type_list=chromosome.battery_type_list,
                )
                for gene in chromosome.genes:
                    mutated_gene = self._mutation_calculation(
                        gene, mutation_params, power_supply_capacity_max
                    )
                    new_chromosome.genes.append(mutated_gene)
                mutated_individual.chromosomes.append(new_chromosome)
            new_offspring.individuals.append(mutated_individual)
        return new_offspring

    def calculate_fitness(self, individual: Individual, resource):
        deadline_violation = 0
        battery_type_violation = 0
        genes_scheduled_dates = set()
        for chromosome in individual.chromosomes:
            chromosome: Chromosome

            for gene in chromosome.genes:
                gene: Gene
                scheduled_date = gene.scheduled_date
                scheduled_date_string = gene.scheduled_date.strftime("%d/%m/%Y")
                battery_type_string = Utils.battery_type_dec_to_str[gene.battery_type]
                power_supply_capacity = gene.power_supply_capacity
                battery_type = gene.battery_type
                time_of_date = gene.time_of_day

                if (battery_type not in chromosome.battery_type_list_num) and (
                    power_supply_capacity > 0
                ):
                    battery_type_violation += 1

                # Check for violation deadline
                if (
                    scheduled_date > gene.end_date and power_supply_capacity > 0
                ) or (
                    scheduled_date < gene.start_date and power_supply_capacity > 0
                ):
                    deadline_violation += 1

                # Tạo một cặp từ scheduled_date và battery_type
                date_battery_pair = (scheduled_date, battery_type_string, time_of_date)
                if power_supply_capacity > 0:
                    if date_battery_pair in genes_scheduled_dates:
                        deadline_violation += 1
                    else:
                        genes_scheduled_dates.add(date_battery_pair)
                # Kiểm tra xem vào thời điểm buổi sáng/chiều trong schedule date có pin cung cấp không
                diesel_morning, diesel_afternoon = Utils.get_diesel_schedule(scheduled_date_string, battery_type_string, resource)
                # Morning
                if time_of_date == 0:
                    if diesel_morning == 0:
                        deadline_violation += 1
                else:
                    # Afternoon
                    if diesel_afternoon == 0:
                        deadline_violation += 1

        individual.deadline_violation = deadline_violation
        individual.battery_type_violation = battery_type_violation

        return deadline_violation, battery_type_violation

    def _crossover_calculation(
        self,
        gen_1: Gene,
        gen_2: Gene,
        crossover_params: CrossoverParams,
        power_supply_capacity_max=int,
    ):

        distri_time_of_day = crossover_params.time_of_day
        distri_diff_date = crossover_params.diff_date
        distri_battery_type = crossover_params.battery_type

        diff_date_gen_1 = (gen_1.scheduled_date - gen_1.start_date).days
        diff_date_gen_2 = (gen_2.scheduled_date - gen_2.start_date).days

        new_diff_date_gen_1, new_diff_date_gen_2 = self._sbx_calculation(
            distri_diff_date, diff_date_gen_1, diff_date_gen_2
        )
        while (
            diff_date_gen_1 > 30
            or diff_date_gen_1 < 0
            or diff_date_gen_2 > 30
            or diff_date_gen_2 < 0
        ):
            diff_date_gen_1 = random.randint(1, 30)
            diff_date_gen_2 = random.randint(1, 30)
            new_diff_date_gen_1, new_diff_date_gen_2 = self._sbx_calculation(
                distri_diff_date, diff_date_gen_1, diff_date_gen_2
            )

        new_time_of_day_gen_1, new_time_of_day_gen_2 = self._sbx_calculation(
            distri_time_of_day, gen_1.time_of_day, gen_2.time_of_day
        )
        while (
            new_time_of_day_gen_1 > 2
            or new_time_of_day_gen_1 < 0
            or new_time_of_day_gen_2 > 2
            or new_time_of_day_gen_2 < 0
        ):
            gen_1.time_of_day = 0
            gen_2.time_of_day = 1
            new_time_of_day_gen_1, new_time_of_day_gen_2 = self._sbx_calculation(
                distri_time_of_day, gen_1.time_of_day, gen_2.time_of_day
            )

        new_battery_type_gen_1, new_battery_type_gen_2 = self._sbx_calculation(
            distri_battery_type, gen_1.battery_type, gen_2.battery_type
        )
        while (
            new_battery_type_gen_1 > 5
            or new_battery_type_gen_1 < 1
            or new_battery_type_gen_2 > 5
            or new_battery_type_gen_2 < 1
        ):
            gen_1.battery_type = random.choice([1, 2, 3, 4, 5])
            gen_2.battery_type = random.choice([1, 2, 3, 4, 5])
            new_battery_type_gen_1, new_battery_type_gen_2 = self._sbx_calculation(
                distri_battery_type, gen_1.battery_type, gen_2.battery_type
            )

        new_gene_1 = Gene(
            gen_1.supply_id,
            gen_1.start_date,
            gen_1.end_date,
            timedelta(days=new_diff_date_gen_1) + gen_1.start_date,
            new_time_of_day_gen_1,
            new_battery_type_gen_1,
            gen_1.power_supply_capacity,
        )
        new_gene_2 = Gene(
            gen_2.supply_id,
            gen_2.start_date,
            gen_2.end_date,
            timedelta(days=new_diff_date_gen_2) + gen_2.start_date,
            new_time_of_day_gen_2,
            new_battery_type_gen_2,
            gen_2.power_supply_capacity,
        )
        return new_gene_1, new_gene_2

    def _sbx_calculation(self, distribution_index: int, v1: int, v2: int):
        random_rate = random.choice(
            [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
        )
        beta = 0
        if random_rate <= 0.5:
            beta = (2 * random_rate) ** (1 / (distribution_index + 1))
        else:
            beta = (1 / (2 - 2 * random_rate)) ** (1 / (distribution_index + 1))

        v1_new = 0.5 * ((1 + beta) * v1 + (1 - beta) * v2)
        v2_new = 0.5 * ((1 - beta) * v1 + (1 + beta) * v2)
        return round(v1_new), round(v2_new)

    def _mutation_calculation(
        self,
        gene: Gene,
        mutation_params: MutationParams,
        power_supply_capacity_max: int,
    ):
        distri_time_of_day = mutation_params.time_of_day
        distri_diff_date = mutation_params.diff_date
        distri_battery_type = mutation_params.battery_type

        new_scheduled_date = self._scheduled_date_mutation(
            distri_diff_date, (gene.scheduled_date - gene.start_date).days
        )
        while new_scheduled_date > 30 or new_scheduled_date < 0:
            diff_date_gen = random.choice([1, 2, 3, 4, 5, 6])
            new_scheduled_date = self._scheduled_date_mutation(
                distri_diff_date, diff_date_gen
            )

        new_battery_type = self._battery_type_mutation(
            distri_battery_type, gene.battery_type
        )
        while new_battery_type > 5 or new_battery_type < 1:
            new_battery_type = self._battery_type_mutation(
                distri_battery_type, random.choice([1, 2, 3, 4, 5])
            )

        new_time_of_day = self._time_of_day_mutation(
            distri_time_of_day, gene.time_of_day
        )
        while new_time_of_day > 2 or new_time_of_day < 0:
            new_time_of_day = self._time_of_day_mutation(
                distri_time_of_day, random.choice([0, 1])
            )

        new_gene = Gene(
            gene.supply_id,
            gene.start_date,
            gene.end_date,
            gene.start_date + timedelta(days=new_scheduled_date),
            new_time_of_day,
            new_battery_type,
            gene.power_supply_capacity,
        )
        return new_gene

    def _scheduled_date_mutation(self, distribution_index: int, scheduled_date: int):
        # if scheduled_date > 30 or scheduled_date < 0:
        #     scheduled_date = random.choice([2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24])
        if scheduled_date > 30 or scheduled_date < 0:
            scheduled_date = random.randint(1, 30)
        random_rate = random.choice(
            [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
        )
        scheduled_date_random = random.choice(
            [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]
        )
        new_scheduled_date = 0
        if random_rate <= 0.5:
            delta = (2 * random_rate) ** (1 / (distribution_index + 1)) - 1
        else:
            delta = 1 - (2 * (1 - random_rate)) ** (1 / (distribution_index + 1))
        if random_rate <= 0.5:
            new_scheduled_date = scheduled_date + delta * (
                scheduled_date - scheduled_date_random
            )
        else:
            new_scheduled_date = scheduled_date + delta * (
                scheduled_date_random - scheduled_date
            )
        return round(new_scheduled_date)

    def _time_of_day_mutation(self, distribution_index: int, time_of_day: int):
        if time_of_day > 2 or time_of_day < 0:
            time_of_day = 0
        random_rate = random.choice(
            [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
        )
        time_of_day_random = random.choice([0, 1])
        new_time_of_day = 0
        if random_rate <= 0.5:
            delta = (2 * random_rate) ** (1 / (distribution_index + 1)) - 1
        else:
            delta = 1 - (2 * (1 - random_rate)) ** (1 / (distribution_index + 1))
        if random_rate <= 0.5:
            new_time_of_day = time_of_day + delta * (time_of_day - time_of_day_random)
        else:
            new_time_of_day = time_of_day + delta * (time_of_day_random - time_of_day)
        return round(new_time_of_day)

    def _battery_type_mutation(self, distribution_index: int, battery_type: int):
        if battery_type > 5 or battery_type < 1:
            battery_type = random.choice([1, 2, 3, 4])
        random_rate = random.choice(
            [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
        )
        battery_type_random = random.choice([1, 2, 3, 4, 5])
        new_battery_type = 0
        if random_rate <= 0.5:
            delta = (2 * random_rate) ** (1 / (distribution_index + 1)) - 1
        else:
            delta = 1 - (2 * (1 - random_rate)) ** (1 / (distribution_index + 1))
        if random_rate <= 0.5:
            new_battery_type = battery_type + delta * (
                battery_type - battery_type_random
            )
        else:
            new_battery_type = battery_type + delta * (
                battery_type_random - battery_type
            )
        return round(new_battery_type)

    def non_dominated_sorting(self, population_size: int, chroms_obj_record: list):
        def dominates(obj1, obj2):
            # Kiểm tra nếu obj1 thống trị obj2.
            better_in_all = all(x <= y for x, y in zip(obj1, obj2))
            strictly_better = any(x < y for x, y in zip(obj1, obj2))
            return better_in_all and strictly_better

        s, n = {}, {}
        front, rank = {}, {}
        front[0] = []

        for p in range(population_size):
            s[p] = []
            n[p] = 0
            for q in range(population_size):
                if dominates(chroms_obj_record[p], chroms_obj_record[q]):
                    s[p].append(q)
                elif dominates(chroms_obj_record[q], chroms_obj_record[p]):
                    n[p] += 1
            if n[p] == 0:
                rank[p] = 0
                front[0].append(p)

        i = 0
        while front[i] != []:
            Q = []
            for p in front[i]:
                for q in s[p]:
                    n[q] -= 1
                    if n[q] == 0:
                        rank[q] = i + 1
                        Q.append(q)
            i += 1
            front[i] = Q

        # Remove the last front if it's empty
        if not front[len(front) - 1]:
            del front[len(front) - 1]

        return front

    def selection(
        self,
        population_size: int,
        front: dict,
        chroms_obj_record: dict,
        total_population: Population,
    ):
        selected_index, selected_obj_dict = (
            self._selection_index_of_population(
                population_size, front, chroms_obj_record
            )
        )

        new_population = Population()
        i = 0
        new_chroms_obj = {}
        for key, (deadline, battery) in selected_obj_dict.items():
            if (
                total_population.individuals[key].deadline_violation == deadline
                and total_population.individuals[key].battery_type_violation == battery
            ):
                new_population.individuals.append(
                    copy.deepcopy(total_population.individuals[key])
                )
                new_chroms_obj[i] = [deadline, battery]
                i = i + 1
            else:
                print("+++++ Error in Selection ++++++++")

        return new_population, new_chroms_obj

    def _selection_index_of_population(
        self,
        population_size: int,
        front: dict,
        chroms_obj_record: List[Tuple[float, float]],
    ) -> Tuple[List[int], Dict[int, Tuple[float, float]]]:
        N = 0
        new_pop = set()  # Use a set to ensure uniqueness of selected points

        # Dictionary to track unique objective points
        unique_points = {}
        for idx, point in enumerate(
            chroms_obj_record.values()
        ):  # Lặp qua giá trị, không phải khóa
            if tuple(point) not in unique_points:
                unique_points[tuple(point)] = (
                    idx  # Track first occurrence index of each unique point
                )

        for i in range(len(front)):
            if N + len(front[i]) > population_size:
                distance = self._calculate_crowding_distance(
                    front[i], chroms_obj_record
                )
                sorted_cdf = sorted(front[i], key=lambda x: distance[x], reverse=True)

                for j in sorted_cdf:
                    # Only add if unique based on objectives
                    if len(new_pop) < population_size and j in unique_points.values():
                        new_pop.add(j)
                    elif len(new_pop) >= population_size:
                        break
                break
            else:
                for j in front[i]:
                    if len(new_pop) < population_size and j in unique_points.values():
                        new_pop.add(j)
                N += len(front[i])

        # Nếu số lượng cá thể trong new_pop chưa đủ, tiếp tục chọn từ các front còn lại
        for i in range(len(front)):
            if len(new_pop) < population_size:
                for j in front[i]:
                    if len(new_pop) < population_size and j in unique_points.values():
                        new_pop.add(j)

        # Convert set back to list for compatibility and create the dictionary with selected points and their objective values
        new_pop = list(new_pop)
        selected_obj_dict = {i: chroms_obj_record[i] for i in new_pop}

        return new_pop, selected_obj_dict

    def _calculate_crowding_distance(
        self, front: List[int], chroms_obj_record: Dict[int, List[float]]
    ) -> Dict[int, float]:
        # Số lượng mục tiêu trong mỗi phần tử của chroms_obj_record
        num_objectives = len(next(iter(chroms_obj_record.values())))
        distance = {i: 0 for i in front}

        for obj_index in range(num_objectives):
            # Sắp xếp các điểm trong mặt trước dựa trên mục tiêu hiện tại
            sorted_front = sorted(front, key=lambda x: chroms_obj_record[x][obj_index])

            # Gán 'inf' cho khoảng cách của các điểm biên
            distance[sorted_front[0]] = float("inf")
            distance[sorted_front[-1]] = float("inf")

            # Tính phạm vi của các giá trị mục tiêu
            objective_values = [chroms_obj_record[i][obj_index] for i in front]
            obj_range = max(objective_values) - min(objective_values)

            # Tính khoảng cách cho các điểm còn lại
            if obj_range > 0:
                for j in range(1, len(sorted_front) - 1):
                    next_obj = chroms_obj_record[sorted_front[j + 1]][obj_index]
                    prev_obj = chroms_obj_record[sorted_front[j - 1]][obj_index]
                    distance[sorted_front[j]] += (next_obj - prev_obj) / obj_range

        return distance
