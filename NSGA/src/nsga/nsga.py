import copy
import random
from datetime import timedelta
from typing import Dict, List, Tuple
import numpy as np
from src.parameters import CrossoverParams, MutationParams
from src.nsga.init import (
    Chromosome,
    Gene,
    Individual,
    Population,
)

from src.nsga.utils import Utils


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
        """Tournament Selection"""
        mating_pool = Population()
        tournament_size = 2

        for _ in range(num_parents_mating):
            tournament = np.random.choice(len(population.individuals), tournament_size, replace=False)
            tournament_individuals = [population.individuals[i] for i in tournament]
            
            # Select the best individual from tournament based on non-domination rank
            best_individual = min(tournament_individuals, 
                                key=lambda x: (x.deadline_violation, x.battery_type_violation))
            mating_pool.individuals.append(copy.deepcopy(best_individual))

        return mating_pool

    def crossover(
        self,
        mating_pool: Population,
        crossover_params: CrossoverParams,
        power_supply_capacity_max: int,
    ):
        offspring = Population()
        parents = copy.deepcopy(mating_pool)
        crossover_rate = 0.9

        while len(parents.individuals) > 0:
            # Randomly select two parents
            mating_idx = np.random.choice(len(parents.individuals), 2, replace=False)
            mating_parents = [
                parents.individuals[mating_idx[0]],
                parents.individuals[mating_idx[1]],
            ]

            individual1: Individual = mating_parents[0]
            individual2: Individual = mating_parents[1]

            if random.random() < crossover_rate:
                new_individual_1 = Individual()
                new_individual_2 = Individual()

                for chromosome_1, chromosome_2 in zip(individual1.chromosomes, individual2.chromosomes):
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

                    # Single point crossover
                    crossover_point = random.randint(0, len(chromosome_1.genes))
                    
                    # Copy genes before crossover point
                    for i in range(crossover_point):
                        new_chromosome_1.genes.append(copy.deepcopy(chromosome_1.genes[i]))
                        new_chromosome_2.genes.append(copy.deepcopy(chromosome_2.genes[i]))
                    
                    # Copy genes after crossover point
                    for i in range(crossover_point, len(chromosome_1.genes)):
                        new_chromosome_1.genes.append(copy.deepcopy(chromosome_2.genes[i]))
                        new_chromosome_2.genes.append(copy.deepcopy(chromosome_1.genes[i]))

                    new_individual_1.chromosomes.append(new_chromosome_1)
                    new_individual_2.chromosomes.append(new_chromosome_2)
            else:
                # If no crossover, copy parents
                new_individual_1 = copy.deepcopy(individual1)
                new_individual_2 = copy.deepcopy(individual2)

            offspring.individuals.append(new_individual_1)
            offspring.individuals.append(new_individual_2)
            parents.individuals = np.delete(parents.individuals, list(mating_idx), axis=0)

        return offspring

    def mutation(
        self,
        offspring_cros: Population,
        mutation_params: MutationParams,
        power_supply_capacity_max: int,
    ):
        offspring = copy.deepcopy(offspring_cros)
        mutation_rate = 0.1

        for individual in offspring.individuals:
            for chromosome in individual.chromosomes:
                for gene in chromosome.genes:
                    if random.random() < mutation_rate:
                        mutation_type = random.random()
                        
                        # Mutate scheduled_date
                        if mutation_type < 0.33:
                            max_days = (gene.end_date - gene.start_date).days
                            new_days = random.randint(0, max_days)
                            gene.scheduled_date = gene.start_date + timedelta(days=new_days)
                        
                        # Mutate time_of_day
                        elif mutation_type < 0.66:
                            gene.time_of_day = random.randint(0, 1)
                        
                        # Mutate battery_type
                        else:
                            valid_battery_types = chromosome.battery_type_list_num
                            if valid_battery_types:
                                gene.battery_type = random.choice(valid_battery_types)

        return offspring

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

                # Check deadline violations
                if (
                    scheduled_date > gene.end_date and power_supply_capacity > 0
                ) or (
                    scheduled_date < gene.start_date and power_supply_capacity > 0
                ):
                    deadline_violation += 1

                # Check for duplicate scheduling
                date_battery_pair = (scheduled_date, battery_type_string, time_of_date)
                if power_supply_capacity > 0:
                    if date_battery_pair in genes_scheduled_dates:
                        deadline_violation += 1
                    else:
                        genes_scheduled_dates.add(date_battery_pair)

                # Check resource availability
                diesel_morning, diesel_afternoon = Utils.get_diesel_schedule(
                    scheduled_date_string, battery_type_string, resource
                )
                if time_of_date == 0:  # Morning
                    if diesel_morning == 0:
                        deadline_violation += 1
                else:  # Afternoon
                    if diesel_afternoon == 0:
                        deadline_violation += 1

        individual.deadline_violation = deadline_violation
        individual.battery_type_violation = battery_type_violation

        return deadline_violation, battery_type_violation

    def non_dominated_sorting(self, population_size: int, chroms_obj_record: dict):
        """
        Thực hiện non-dominated sorting theo NSGA-II thuần túy
        """
        fronts = {}  # Dictionary lưu các front
        fronts[0] = []  # Front đầu tiên
        
        # Khởi tạo các tập S và n cho mỗi cá thể
        S = {}  # Tập các cá thể bị p thống trị
        n = {}  # Số lượng cá thể thống trị p
        rank = {}  # Rank của mỗi cá thể
        
        # Khởi tạo cho mỗi cá thể
        for p in range(population_size):
            S[p] = []
            n[p] = 0
            
            # So sánh với các cá thể khác
            for q in range(population_size):
                if p != q:
                    if self._dominates(chroms_obj_record[p], chroms_obj_record[q]):
                        S[p].append(q)
                    elif self._dominates(chroms_obj_record[q], chroms_obj_record[p]):
                        n[p] += 1
            
            # Nếu không bị ai thống trị, thuộc front 0
            if n[p] == 0:
                rank[p] = 0
                fronts[0].append(p)
        
        # Tìm các front tiếp theo
        i = 0
        current_front = fronts[0]
        
        while current_front:
            next_front = []
            for p in current_front:
                for q in S[p]:
                    n[q] -= 1
                    if n[q] == 0:
                        rank[q] = i + 1
                        next_front.append(q)
            i += 1
            if next_front:
                fronts[i] = next_front
                current_front = next_front
            else:
                break
                
        return fronts

    def _dominates(self, obj1, obj2):
        """
        Kiểm tra obj1 có thống trị obj2 không trong bài toán tối thiểu hóa
        """
        at_least_one_better = False
        for val1, val2 in zip(obj1, obj2):
            if val1 > val2:  # obj1 tệ hơn obj2
                return False
            elif val1 < val2:  # obj1 tốt hơn obj2 trong ít nhất một mục tiêu
                at_least_one_better = True
        return at_least_one_better

    def calculate_crowding_distance(self, front: List[int], chroms_obj_record: Dict[int, List[float]]) -> Dict[int, float]:
        """Calculate crowding distance for solutions in a front"""
        distance = {i: 0 for i in front}
        
        if len(front) <= 2:
            for i in front:
                distance[i] = float('inf')
            return distance

        for obj_index in range(2):  # For each objective
            # Sort front by objective value
            sorted_front = sorted(front, key=lambda x: chroms_obj_record[x][obj_index])
            
            # Set infinite distance to boundary points
            distance[sorted_front[0]] = float('inf')
            distance[sorted_front[-1]] = float('inf')
            
            # Calculate crowding distance
            obj_range = (
                chroms_obj_record[sorted_front[-1]][obj_index] - 
                chroms_obj_record[sorted_front[0]][obj_index]
            )
            
            if obj_range == 0:
                continue
                
            for i in range(1, len(sorted_front) - 1):
                distance[sorted_front[i]] += (
                    chroms_obj_record[sorted_front[i + 1]][obj_index] - 
                    chroms_obj_record[sorted_front[i - 1]][obj_index]
                ) / obj_range

        return distance

    def selection(
        self,
        population_size: int,
        front: dict,
        chroms_obj_record: dict,
        total_population: Population,
    ):
        """Select the best solutions using non-dominated sorting and crowding distance"""
        new_population = Population()
        new_chroms_obj = {}
        current_count = 0
        front_num = 0
        
        while current_count < population_size and front_num < len(front):
            current_front = front[front_num]
            
            if current_count + len(current_front) <= population_size:
                # Add all solutions from current front
                for idx in current_front:
                    new_population.individuals.append(copy.deepcopy(total_population.individuals[idx]))
                    new_chroms_obj[current_count] = chroms_obj_record[idx]
                    current_count += 1
            else:
                # Calculate crowding distance for current front
                crowding_distances = self.calculate_crowding_distance(current_front, chroms_obj_record)
                
                # Sort solutions by crowding distance
                sorted_front = sorted(current_front, 
                                   key=lambda x: crowding_distances[x],
                                   reverse=True)
                
                # Add solutions until population is filled
                for idx in sorted_front:
                    if current_count >= population_size:
                        break
                    new_population.individuals.append(copy.deepcopy(total_population.individuals[idx]))
                    new_chroms_obj[current_count] = chroms_obj_record[idx]
                    current_count += 1
            
            front_num += 1

        return new_population, new_chroms_obj