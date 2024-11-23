import copy
import random
from datetime import timedelta
from typing import Dict, List, Tuple
import numpy as np
from src.parameters import CrossoverParams, MutationParams
from src.pmsbx_ga.init import (
    Chromosome,
    Gene,
    Individual,
    Population,
)
from src.all_packages import PATHS
from src.pmsbx_ga.utils import Utils


class GeneticOperators:

    def __init__(self, supply_orders=None, session_capacity_threshold=None, population_size=0):
        if (
            supply_orders is not None
            and not supply_orders.empty
            and population_size > 0
        ):
            self.population = Population(supply_orders, session_capacity_threshold, population_size)
            self.resource = Utils.load_data(PATHS["resource"])
        else:
            self.population = []
            self.resource = None

    def select_mating_pool(self, population: Population, num_parents_mating: int):
        mating_pool = Population()
        tournament_size = 5
        
        for _ in range(num_parents_mating):
            # Chọn ngẫu nhiên tournament_size cá thể
            tournament = np.random.choice(population.individuals, tournament_size, replace=False)
            
            # Chọn cá thể có fitness tốt nhất từ tournament
            winner = min(tournament, key=lambda ind: self.calculate_fitness(ind, self.resource))
            mating_pool.individuals.append(copy.deepcopy(winner))
            
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

        # Tính fitness tổng hợp (có thể điều chỉnh trọng số w1, w2)
        w1 = 0.6  # trọng số cho deadline_violation 
        w2 = 0.4  # trọng số cho battery_type_violation
        total_fitness = w1 * deadline_violation + w2 * battery_type_violation
        
        return total_fitness

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
