import pandas as pd
from src.all_packages import PATHS
from src.parameters import CrossoverParams, IntervalParams, MutationParams
from src.save_load_data import PopulationSerializer
from src.nsga.utils import Utils
from chart.draw_chart import save_evolution_charts
from src.nsga.init import Population
from src.nsga import nsga  # Thay đổi import từ pmsbx_nsga sang nsga
import numpy as np
import copy  # Add this import
class NsgaAlgorithm:  # Đổi tên class để phản ánh việc sử dụng NSGA-II thuần túy

    def __init__(
        self,
        popsize: int,
        num_parents_mating: int,
        power_supply_capacity_max: int,
        session_capacity_threshold: int,
        interval_params: IntervalParams,  # Bỏ crossover_params và mutation_params vì không cần thiết
        num_generations: int,
        is_load_data: bool,
    ):
        self.popsize = popsize
        self.num_parents_mating = num_parents_mating
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
        # Initialize tracking variables
        best_fitness = float('inf')
        best_individual = None
        all_generations_data = []
        fitness_history = []
        SAVE_INTERVAL = 50

        if self.is_load_data == False:
            nsga_algorithm = nsga.GeneticOperators(
                self.supply_orders, self.session_capacity_threshold, self.popsize
            )
        else:
            nsga_algorithm = nsga.GeneticOperators(self.supply_orders)

            nsga_algorithm.population = PopulationSerializer.load_population_from_json(
                "./save_data/population_data.json"
            )
            chroms_obj_record = PopulationSerializer.load_chroms_obj_record_from_json(
                "./save_data/chroms_obj_record.json"
            )
            front = nsga_algorithm.non_dominated_sorting(
                len(nsga_algorithm.population.individuals), chroms_obj_record
            )

            new_population, new_chroms_obj = nsga_algorithm.selection(
                self.popsize, front, chroms_obj_record, nsga_algorithm.population
            )
            nsga_algorithm.population = new_population

        # Calculate initial fitness
        for i, individual in enumerate(nsga_algorithm.population.individuals):
            deadline_violation, battery_type_violation = nsga_algorithm.calculate_fitness(
                individual, self.resource
            )
            self.chroms_obj[i] = [deadline_violation, battery_type_violation]

        # Open file for fitness history
        with open("fitness_history.txt", "w") as file:
            for index in range(self.num_generations):
                prev_best_fitness = best_fitness

                # Debug population statistics every 10 generations
                if index % 10 == 0:
                    population_fitnesses = [sum(self.chroms_obj[i]) for i in range(len(nsga_algorithm.population.individuals))]
                    print(f"\nGeneration {index} Population Stats:")
                    print(f"Mean Fitness: {sum(population_fitnesses)/len(population_fitnesses):.2f}")
                    print(f"Best Fitness: {min(population_fitnesses):.2f}")
                    print(f"Worst Fitness: {max(population_fitnesses):.2f}")
                    print(f"Fitness Variance: {np.var(population_fitnesses):.2f}")
                    print(f"""
                        Generation {index}:
                        - Current Best Fitness: {min(population_fitnesses)}
                        - Overall Best Fitness: {best_fitness}
                        - Improvement: {prev_best_fitness - best_fitness if prev_best_fitness != float('inf') else 0}
                    """)

                    # Export violations for best individual
                    best_idx = np.argmin(population_fitnesses)
                    best_ind = nsga_algorithm.population.individuals[best_idx]
                    nsga_algorithm.export_violations(best_ind, index)

                    # Export Pareto front
                    nsga_algorithm.export_pareto_front(nsga_algorithm.population, index)
                
                # Selection
                parents = nsga_algorithm.select_mating_pool(nsga_algorithm.population, self.num_parents_mating)
                
                # Crossover
                offspring_crossover = nsga_algorithm.crossover(
                    parents, None, self.power_supply_capacity_max  # None for crossover_params
                )
                
                # Mutation
                offspring_mutation = nsga_algorithm.mutation(
                    offspring_crossover, None, self.power_supply_capacity_max  # None for mutation_params
                )

                # Combine populations
                total_population = Population()
                total_population.individuals.extend(nsga_algorithm.population.individuals)
                total_population.individuals.extend(offspring_crossover.individuals)
                total_population.individuals.extend(offspring_mutation.individuals)

                # Calculate fitness for new individuals
                chroms_obj_record = self.chroms_obj.copy()
                for i, individual in enumerate(total_population.individuals[self.popsize:], start=self.popsize):
                    deadline_violation, battery_type_violation = nsga_algorithm.calculate_fitness(
                        individual, self.resource
                    )
                    chroms_obj_record[i] = [deadline_violation, battery_type_violation]

                # Track best solution
                for i, individual in enumerate(total_population.individuals):
                    current_fitness = sum(chroms_obj_record[i])
                    if current_fitness < best_fitness:
                        best_fitness = current_fitness
                        best_individual = copy.deepcopy(individual)

                # Save population if needed
                if index != 0 and index % self.interval_save_population == 0:
                    print(f"******** Save population -> Iteration: {index} **********")
                    PopulationSerializer.save_population_to_json(
                        total_population, "./save_data/population_data.json"
                    )
                    PopulationSerializer.save_chroms_obj_record_to_json(
                        chroms_obj_record, "./save_data/chroms_obj_record.json"
                    )

                # Save best solution periodically
                if index % 100 == 0:
                    best_population = Population()
                    best_population.individuals = [best_individual]
                    PopulationSerializer.save_population_to_json(
                        best_population, f"./save_data/best_solution_gen_{index}.json"
                    )

                # Non-dominated sorting and selection
                front = nsga_algorithm.non_dominated_sorting(
                    len(total_population.individuals), chroms_obj_record
                )

                new_population, new_chroms_obj = nsga_algorithm.selection(
                    self.popsize, front, chroms_obj_record, total_population
                )
                nsga_algorithm.population = new_population
                self.chroms_obj = new_chroms_obj

                # Save generation data for visualization
                if index % SAVE_INTERVAL == 0:
                    generation_data = list(new_chroms_obj.values())
                    all_generations_data.append((index, generation_data))
                
                # Write current best fitness to history
                file.write(f"{min([sum(v) for v in new_chroms_obj.values()])}\n")
                fitness_history.append(min([sum(v) for v in new_chroms_obj.values()]))

                # Save Pareto points
                if (index) % self.interval_save_pareto == 0 or index == 0:
                    points = []
                    for key, value in new_chroms_obj.items():
                        points.append(value)
                    with open("pareto_points.txt", "a") as pareto_file:
                        pareto_file.write(f"{points}\n")

        # Save final visualization
        save_evolution_charts(all_generations_data, fitness_history, self.num_generations)