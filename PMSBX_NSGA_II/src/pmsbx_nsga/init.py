from datetime import datetime, timedelta
import random

import numpy as np
import pandas as pd

from src.pmsbx_nsga.utils import Utils


# Class representing a single gene
class Gene:
    def __init__(
        self,
        supply_id,
        start_date,
        end_date,
        scheduled_date,
        time_of_day,
        battery_type,
        power_supply_capacity,
    ):
        self.supply_id = supply_id
        self.start_date = start_date
        self.end_date = end_date
        self.scheduled_date = scheduled_date
        self.time_of_day = time_of_day
        self.battery_type = battery_type
        self.power_supply_capacity = power_supply_capacity


class Chromosome:

    def __init__(
        self,
        supply_order=None,
        session_capacity_threshold=None,
        num_genes_per_chromo=0,
        total_expected=None,
        device_type=None,
        battery_type_list=None,
    ):
        if (
            supply_order is not None
            and not supply_order.empty
            and num_genes_per_chromo > 0
        ):
            self.total_expected = supply_order["total_expected"]
            self.device_type = supply_order["device_type"]
            self.battery_type_list = supply_order["battery_type"].split("|")
            self.battery_type_list_num = Utils.covert_battery_type_list_to_numbers(
                self.battery_type_list
            )
            self.genes = self._generate_genes(
                supply_order, session_capacity_threshold, num_genes_per_chromo
            )
        elif (
            total_expected is not None
            and device_type is not None
            and battery_type_list is not None
        ):
            self.total_expected = total_expected
            self.device_type = device_type
            self.battery_type_list = battery_type_list
            self.battery_type_list_num = Utils.covert_battery_type_list_to_numbers(
                self.battery_type_list
            )
            self.genes = []

    def _generate_genes(self, supply_order, session_capacity_threshold, num_genes_per_chromo):
        genes = []

        supply_id = supply_order["supply_id"]
        priority = supply_order["priority"]
        device_type = supply_order["device_type"]
        start_date = supply_order["start_day"]
        end_date = supply_order["end_date"]
        end_date = supply_order["end_date"]

        num_genes_per_chromo = 4
        amount_to_supply = self.total_expected

        for _ in range(num_genes_per_chromo):
            start, end, scheduled = self._random_scheduled_date(start_date, end_date)
            time_of_day = random.choice([0, 1])
            battery_type = random.choice(self.battery_type_list_num)
            power_supply_capacity = session_capacity_threshold
            if amount_to_supply - power_supply_capacity < 0:
                power_supply_capacity = 0
                amount_to_supply = 0
            else:
                amount_to_supply = amount_to_supply - power_supply_capacity

            gene = Gene(
                supply_id,
                start,
                end,
                scheduled,
                time_of_day,
                battery_type,
                power_supply_capacity,
            )
            genes.append(gene)

        return genes

    def _random_scheduled_date(self, start_date, end_date):
        start = datetime.strptime(start_date, "%d/%m/%Y").date()
        end = datetime.strptime(end_date, "%d/%m/%Y").date()
        delta = end - start
        random_days = random.randint(0, delta.days)
        scheduled = start + timedelta(days=random_days)
        return start, end, scheduled


# Class representing an individual which is a collection of chromosomes
class Individual:
    def __init__(self, supply_orders=None, session_capacity_threshold=None, num_genes_per_chromo=0):
        self.deadline_violation = 0
        self.battery_type_violation = 0
        if (
            supply_orders is not None
            and not supply_orders.empty
            and num_genes_per_chromo > 0
        ):
            self.chromosomes = self._init_individual(
                supply_orders, session_capacity_threshold, num_genes_per_chromo
            )
        else:
            self.chromosomes = []

    def _init_individual(self, supply_orders, session_capacity_threshold, num_genes_per_chromo):
        chromosomes = []
        for _, supply_order in supply_orders.iterrows():
            chromosome = Chromosome(supply_order, session_capacity_threshold, num_genes_per_chromo)
            chromosomes.append(chromosome)
        return chromosomes


# Class managing the population of individuals
class Population:
    def __init__(self, supply_orders=None, session_capacity_threshold= None, population_size=0):
        if (
            supply_orders is not None
            and not supply_orders.empty
            and population_size > 0
        ):
            self.individuals = self._init_population(
                supply_orders, session_capacity_threshold, population_size
            )
        else:
            self.individuals = []

    def _init_population(self, supply_orders, session_capacity_threshold, population_size):
        num_genes_per_chromo = Utils.calculate_num_genes_per_chromo(
            supply_orders, session_capacity_threshold
        )
        return [
            Individual(supply_orders, session_capacity_threshold, num_genes_per_chromo)
            for _ in range(population_size)
        ]
