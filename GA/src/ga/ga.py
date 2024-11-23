import copy
import random
from datetime import timedelta
from typing import Dict, List, Tuple
import numpy as np
from src.parameters import GeneticParams
from src.ga.init import (
    Chromosome,
    Gene,
    Individual,
    Population,
)
from src.all_packages import PATHS
from src.ga.utils import Utils


class GeneticOperators:
    def __init__(self, supply_orders=None, session_capacity_threshold=None, population_size=0):
        if (supply_orders is not None and not supply_orders.empty and population_size > 0):
            self.population = Population(supply_orders, session_capacity_threshold, population_size)
            self.resource = Utils.load_data(PATHS["resource"])
        else:
            self.population = []
            self.resource = None

    def select_mating_pool(self, population: Population, num_parents_mating: int):
        mating_pool = Population()
        tournament_size = 5
        
        for _ in range(num_parents_mating):
            tournament = np.random.choice(population.individuals, tournament_size, replace=False)
            winner = min(tournament, key=lambda ind: self.calculate_fitness(ind, self.resource))
            mating_pool.individuals.append(copy.deepcopy(winner))
            
        return mating_pool

    def crossover(self, parents: Population, crossover_rate: float, power_supply_capacity_max: int):
        offspring = Population()
        
        while len(parents.individuals) > 1:
            idx1, idx2 = np.random.choice(len(parents.individuals), 2, replace=False)
            parent1 = parents.individuals[idx1]
            parent2 = parents.individuals[idx2]
            
            if random.random() < crossover_rate:
                child1, child2 = self._one_point_crossover(parent1, parent2)
                offspring.individuals.extend([child1, child2])
            else:
                offspring.individuals.extend([copy.deepcopy(parent1), copy.deepcopy(parent2)])
                
            parents.individuals = np.delete(parents.individuals, [idx1, idx2], axis=0)
        
        return offspring

    def mutation(self, offspring: Population, mutation_rate: float, power_supply_capacity_max: int):
        for individual in offspring.individuals:
            for chromosome in individual.chromosomes:
                for gene in chromosome.genes:
                    if random.random() < mutation_rate:
                        # Simple random mutations for each gene component
                        gene.time_of_day = random.choice([0, 1])
                        
                        # Đảm bảo battery_type là số nguyên từ 1-5
                        if isinstance(gene.battery_type, str):
                            gene.battery_type = Utils.battery_type_str_to_dec.get(gene.battery_type, 1)
                        # Chọn ngẫu nhiên từ danh sách số
                        gene.battery_type = random.choice([1, 2, 3, 4, 5])
                        
                        max_days = (gene.end_date - gene.start_date).days
                        random_days = random.randint(0, max_days)
                        gene.scheduled_date = gene.start_date + timedelta(days=random_days)
                        gene.power_supply_capacity = random.randint(1, power_supply_capacity_max)
        
        return offspring

    def calculate_fitness(self, individual: Individual, resource):
        deadline_violation = 0
        battery_type_violation = 0
        genes_scheduled_dates = set()
        
        for chromosome in individual.chromosomes:
            for gene in chromosome.genes:
                scheduled_date = gene.scheduled_date
                scheduled_date_string = gene.scheduled_date.strftime("%d/%m/%Y")
                battery_type_string = Utils.battery_type_dec_to_str[gene.battery_type]
                power_supply_capacity = gene.power_supply_capacity
                battery_type = gene.battery_type
                time_of_date = gene.time_of_day

                # Check battery type violation
                if (battery_type not in chromosome.battery_type_list_num) and (power_supply_capacity > 0):
                    battery_type_violation += 1

                # Check deadline violation
                if ((scheduled_date > gene.end_date or scheduled_date < gene.start_date) 
                    and power_supply_capacity > 0):
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
                if time_of_date == 0 and diesel_morning == 0:
                    deadline_violation += 1
                elif time_of_date == 1 and diesel_afternoon == 0:
                    deadline_violation += 1

        individual.deadline_violation = deadline_violation
        individual.battery_type_violation = battery_type_violation
        
        # Calculate weighted fitness
        w1 = 0.6  # weight for deadline violation
        w2 = 0.4  # weight for battery type violation
        total_fitness = w1 * deadline_violation + w2 * battery_type_violation
        
        return total_fitness

    def _one_point_crossover(self, parent1: Individual, parent2: Individual):
        child1 = copy.deepcopy(parent1)
        child2 = copy.deepcopy(parent2)
        
        for chrom1, chrom2 in zip(child1.chromosomes, child2.chromosomes):
            if len(chrom1.genes) > 0:
                crossover_point = random.randint(0, len(chrom1.genes)-1)
                chrom1.genes[crossover_point:], chrom2.genes[crossover_point:] = \
                    chrom2.genes[crossover_point:], chrom1.genes[crossover_point:]
                    
        return child1, child2