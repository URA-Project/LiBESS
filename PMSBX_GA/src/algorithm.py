import pandas as pd
from src.all_packages import PATHS
from src.parameters import CrossoverParams, MutationParams
from src.save_load_data import PopulationSerializer
from src.pmsbx_ga.utils import Utils
from src.pmsbx_ga.init import Population
from src.pmsbx_ga import pmsbx_ga
import copy
import random
import numpy as np
from chart.draw_chart import save_evolution_charts


class PmsbxGaAlgorithm:

    def __init__(
        self,
        popsize: int,
        num_parents_mating: int,
        power_supply_capacity_max: int,
        session_capacity_threshold: int,
        crossover_params: CrossoverParams,
        mutation_params: MutationParams,
        num_generations: int,
        is_load_data: bool,
        tournament_size: int = 3,
    ):
        self.popsize = popsize
        self.num_parents_mating = num_parents_mating
        self.mutation_params = mutation_params
        self.crossover_params = crossover_params
        self.num_generations = num_generations
        self.power_supply_capacity_max = power_supply_capacity_max
        self.session_capacity_threshold = session_capacity_threshold
        self.supply_orders = pd.read_csv(PATHS["supply_orders"])
        required_columns = ["supply_id", "priority", "start_day", "end_date", "total_expected", "device_type", "battery_type"]
        
        missing_columns = [col for col in required_columns 
                        if col not in self.supply_orders.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns in CSV: {missing_columns}")
        self.resource = Utils.load_data(PATHS["resource"])
        self.population = []
        self.is_load_data = is_load_data
        self.chroms_obj = {}  # Khởi tạo dictionary để lưu pareto points
        self.tournament_size = tournament_size
    def run_algorithm(self):
        # Khởi tạo lists để lưu dữ liệu
        all_generations_data = []  # Lưu data cho pareto front
        fitness_history = []      # Lưu fitness history
        
        # Lưu dữ liệu mỗi 50 generations thay vì mọi generation
        SAVE_INTERVAL = 50
        
        if not self.is_load_data:
            ga = pmsbx_ga.GeneticOperators(
                self.supply_orders, self.session_capacity_threshold, self.popsize
            )
            
            # Thu thập dữ liệu ban đầu
            for i, individual in enumerate(ga.population.individuals):
                ga.calculate_fitness(individual, self.resource)
                self.chroms_obj[i] = ga.calculate_fitness(individual, self.resource)
                
        else:
            ga = pmsbx_ga.GeneticOperators(self.supply_orders)
            ga.population = PopulationSerializer.load_population_from_json(
                "./save_data/population_data.json"
            )

        best_fitness = float('inf')
        best_individual = None
        
        # Mở file để ghi fitness history
        with open("fitness_history.txt", "w") as file:
            # Thêm list để lưu pareto points
            self.chroms_obj = {}  # dictionary để lưu pareto points
            
            for generation in range(self.num_generations):
                prev_best_fitness = best_fitness
                
                # Debug population diversity
                population_fitnesses = [ga.calculate_fitness(ind, self.resource) 
                                      for ind in ga.population.individuals]
                print(f"\nGeneration {generation} Population Stats:")
                print(f"Mean Fitness: {sum(population_fitnesses)/len(population_fitnesses):.2f}")
                print(f"Best Fitness: {min(population_fitnesses):.2f}")
                print(f"Worst Fitness: {max(population_fitnesses):.2f}")
                print(f"Fitness Variance: {np.var(population_fitnesses):.2f}")

                # Selection with debug
                parents = ga.select_mating_pool(ga.population, self.num_parents_mating)
                parent_fitnesses = [ga.calculate_fitness(p, self.resource) for p in parents.individuals]
                print("\nSelected Parents Stats:")
                print(f"Mean Parent Fitness: {sum(parent_fitnesses)/len(parent_fitnesses):.2f}")
                print(f"Best Parent: {min(parent_fitnesses):.2f}")
                
                # Crossover & Mutation với debug
                offspring_crossover = ga.crossover(parents, self.crossover_params, 
                                                self.power_supply_capacity_max)
                offspring_mutation = ga.mutation(offspring_crossover, self.mutation_params,
                                              self.power_supply_capacity_max)
                
                offspring_fitnesses = [ga.calculate_fitness(ind, self.resource) 
                                     for ind in offspring_mutation.individuals]
                print("\nOffspring Stats:")
                print(f"Mean Offspring Fitness: {sum(offspring_fitnesses)/len(offspring_fitnesses):.2f}")
                print(f"Best Offspring: {min(offspring_fitnesses):.2f}")
                
                # Evaluate all individuals
                all_population = Population()
                current_best_fitness = float('inf')
                
                # Evaluate current population
                for individual in ga.population.individuals:
                    fitness = ga.calculate_fitness(individual, self.resource)
                    if fitness < current_best_fitness:
                        current_best_fitness = fitness
                    if fitness < best_fitness:
                        best_fitness = fitness
                        best_individual = copy.deepcopy(individual)
                    all_population.individuals.append(individual)
                
                # Evaluate offspring
                for individual in offspring_mutation.individuals:
                    fitness = ga.calculate_fitness(individual, self.resource)
                    if fitness < current_best_fitness:
                        current_best_fitness = fitness
                    if fitness < best_fitness:
                        best_fitness = fitness
                        best_individual = copy.deepcopy(individual)
                    all_population.individuals.append(individual)

                # Create new population
                new_population = Population()
                
                # Elitism - keep top N individuals
                num_elites = 5
                if best_individual is not None:
                    sorted_population = sorted(all_population.individuals, 
                                             key=lambda x: ga.calculate_fitness(x, self.resource))
                    for i in range(min(num_elites, len(sorted_population))):
                        new_population.individuals.append(copy.deepcopy(sorted_population[i]))
                
                # Tournament selection với debug
                tournament_winners = []
                while len(new_population.individuals) < self.popsize:
                    tournament_size = self.tournament_size
                    tournament = random.sample(all_population.individuals, tournament_size)
                    tournament_fitnesses = [ga.calculate_fitness(ind, self.resource) for ind in tournament]
                    
                    winner = min(tournament, key=lambda x: ga.calculate_fitness(x, self.resource))
                    winner_fitness = ga.calculate_fitness(winner, self.resource)
                    tournament_winners.append(winner_fitness)
                    new_population.individuals.append(copy.deepcopy(winner))
                
                print(f"\nTournament Winners Stats:")
                print(f"Mean Winner Fitness: {sum(tournament_winners)/len(tournament_winners):.2f}")
                print(f"Best Winner: {min(tournament_winners):.2f}")
                
                # Debug new population formation
                print("\nNew Population Formation:")
                print(f"Elite Individual Fitness: {ga.calculate_fitness(best_individual, self.resource):.2f}")
                
                # Update population
                ga.population = new_population
                
                # Save current generation's best fitness
                file.write(f"{current_best_fitness}\n")
                fitness_history.append(current_best_fitness)
                
                # Save best solution periodically
                if generation % 100 == 0:
                    PopulationSerializer.save_population_to_json(
                        ga.population, f"./save_data/population_gen_{generation}.json"
                    )
                
                # Log detailed information every 10 generations
                if generation % 10 == 0:
                    print(f"""
                        Generation {generation}:
                        - Current Best Fitness: {current_best_fitness}
                        - Overall Best Fitness: {best_fitness}
                        - Improvement: {prev_best_fitness - best_fitness if prev_best_fitness != float('inf') else 0}
                        """)
                
                # Cập nhật chroms_obj sau mỗi generation
                self.chroms_obj.clear()  # Xóa dữ liệu cũ
                for i, individual in enumerate(ga.population.individuals):
                    ga.calculate_fitness(individual, self.resource)
                    self.chroms_obj[i] = ga.calculate_fitness(individual, self.resource)
                
                # Lưu pareto data mỗi 50 generations
                if generation % SAVE_INTERVAL == 0:
                    generation_data = list(self.chroms_obj.values())
                    all_generations_data.append((generation, generation_data))
        
        # Sau khi kết thúc vòng lặp generations
        # Vẽ và lưu biểu đồ
        save_evolution_charts(all_generations_data, fitness_history, self.num_generations)
        
        # Save final best solution
        PopulationSerializer.save_population_to_json(
            Population([best_individual]), "./save_data/best_solution.json"
        )
