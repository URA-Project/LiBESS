import pandas as pd
from src.all_packages import PATHS
from src.parameters import CrossoverParams, IntervalParams, MutationParams
from src.save_load_data import PopulationSerializer
from src.pmsbx_nsga.utils import Utils
from src.pmsbx_nsga.init import Population
from src.pmsbx_nsga import pmsbx_nsga


class PmsbxNsgaAlgorithm:

    def __init__(
        self,
        popsize: int,
        num_parents_mating: int,
        power_supply_capacity_max: int,
        session_capacity_threshold: int,
        crossover_params: CrossoverParams,
        mutation_params: MutationParams,
        interval_params: IntervalParams,
        num_generations: int,
        is_load_data: bool,
    ):
        self.popsize = popsize
        self.num_parents_mating = num_parents_mating
        self.mutation_params = mutation_params
        self.crossover_params = crossover_params
        self.num_generations = num_generations
        self.power_supply_capacity_max = power_supply_capacity_max
        self.session_capacity_threshold = session_capacity_threshold
        self.interval_save_pareto = interval_params.save_pareto
        self.interval_save_population = interval_params.save_population
        self.supply_orders = pd.read_csv(PATHS["supply_orders"])
        self.resource = Utils.load_data(PATHS["resource"])
        self.population = []
        self.is_load_data = is_load_data
        self.chroms_obj = {}

    def run_algorithm(self):
        if self.is_load_data == False:
            nsga = pmsbx_nsga.GeneticOperators(
                self.supply_orders, self.session_capacity_threshold, self.popsize
            )
        else:
            nsga = pmsbx_nsga.GeneticOperators(self.supply_orders)

            nsga.population = PopulationSerializer.load_population_from_json(
                "./save_data/population_data.json"
            )
            chroms_obj_record = PopulationSerializer.load_chroms_obj_record_from_json(
                "./save_data/chroms_obj_record.json"
            )
            front = nsga.non_dominated_sorting(
                len(nsga.population.individuals), chroms_obj_record
            )

            """----------selection----------"""
            new_population, new_chroms_obj = nsga.selection(
                self.popsize, front, chroms_obj_record, nsga.population
            )
            nsga.population = []
            nsga.population = new_population

        # Lưu pareto_points cho lần đầu khởi tạo
        for i, individual in enumerate(nsga.population.individuals):
            battery_type_violation, deadline_violation = nsga.calculate_fitness(
                individual, self.resource
            )
            self.chroms_obj[i] = [battery_type_violation, deadline_violation]

        for index in range(self.num_generations):
            if index % 10 == 0:
                print(f"******** Iteration : {index} **********")
            parents = nsga.select_mating_pool(nsga.population, self.num_parents_mating)
            offspring_crossover = nsga.crossover(
                parents, self.crossover_params, self.power_supply_capacity_max
            )
            offspring_mutation = nsga.mutation(
                offspring_crossover,
                self.mutation_params,
                self.power_supply_capacity_max,
            )
            parents_mutation = nsga.mutation(
                parents, self.mutation_params, self.power_supply_capacity_max
            )

            total_population = Population()

            for individual in nsga.population.individuals:
                total_population.individuals.append(individual)

            for individual in offspring_crossover.individuals:
                total_population.individuals.append(individual)

            for individual in offspring_mutation.individuals:
                total_population.individuals.append(individual)

            for individual in parents_mutation.individuals:
                total_population.individuals.append(individual)

            chroms_obj_record = self.chroms_obj
            for i, individual in enumerate(total_population.individuals[self.popsize:], start=self.popsize):
                battery_type_violation, deadline_violation = nsga.calculate_fitness(
                    individual, self.resource
                )
                chroms_obj_record[i] = [battery_type_violation, deadline_violation]

            if index != 0 and index % self.interval_save_population == 0:
                print(f"******** Save population -> Iteration: {index} **********")
                PopulationSerializer.save_population_to_json(
                    total_population, "./save_data/population_data.json"
                )
                PopulationSerializer.save_chroms_obj_record_to_json(
                    chroms_obj_record, "./save_data/chroms_obj_record.json"
                )

            front = nsga.non_dominated_sorting(
                len(total_population.individuals), chroms_obj_record
            )

            new_population, new_chroms_obj = nsga.selection(
                self.popsize, front, chroms_obj_record, total_population
            )
            nsga.population = []
            nsga.population = new_population
            self.chroms_obj = new_chroms_obj

            """---------- save pareto in new population ----------"""
            if (index) % self.interval_save_pareto == 0 or index == 0:
                points = []
                for key, value in new_chroms_obj.items():
                    points.append(value)
                with open("pareto_points.txt", "a") as file:
                    file.write(f"{points}\n")
