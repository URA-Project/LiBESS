import pandas as pd
import numpy as np
import copy
import random
import time
from src.all_packages import PATHS
from src.parameters import GeneticParams
from src.save_load_data import PopulationSerializer
from src.ga.utils import Utils
from src.ga.init import Population
from src.ga import ga
from chart.draw_chart import save_evolution_charts

class GaAlgorithm:
    def __init__(
        self,
        popsize: int,
        num_parents_mating: int,
        power_supply_capacity_max: int,
        session_capacity_threshold: int,
        genetic_params: GeneticParams,
        num_generations: int,
        is_load_data: bool,
        tournament_size: int = 3,
        num_elites: int = 5
    ):
        self.popsize = popsize
        self.num_parents_mating = num_parents_mating
        self.genetic_params = genetic_params
        self.num_generations = num_generations
        self.power_supply_capacity_max = power_supply_capacity_max
        self.session_capacity_threshold = session_capacity_threshold
        self.supply_orders = pd.read_csv(PATHS["supply_orders"])
        self.num_elites = num_elites
        
        required_columns = ["supply_id", "priority", "start_day", "end_date", 
                          "total_expected", "device_type", "battery_type"]
        missing_columns = [col for col in required_columns 
                        if col not in self.supply_orders.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns in CSV: {missing_columns}")
            
        self.resource = Utils.load_data(PATHS["resource"])
        self.population = []
        self.is_load_data = is_load_data
        self.tournament_size = tournament_size

    def _evaluate_population(self, population, ga_operator):
        """Helper method to evaluate entire population"""
        return [ga_operator.calculate_fitness(ind, self.resource) 
                for ind in population.individuals]

    def run_algorithm(self):
        start_time = time.time()
        fitness_history = []
        
        # Initialize GA operator
        ga_operator = ga.GeneticOperators(
            self.supply_orders, 
            self.session_capacity_threshold,
            self.popsize
        )
        
        # Initial population evaluation
        all_fitnesses = [ga_operator.calculate_fitness(ind, self.resource) 
                        for ind in ga_operator.population.individuals]
        best_fitness = min(all_fitnesses)
        best_individual = copy.deepcopy(
            ga_operator.population.individuals[np.argmin(all_fitnesses)]
        )
        generations_without_improvement = 0
        
        # Mở file để ghi fitness history
        with open("fitness_history.txt", "w") as file:
            for generation in range(self.num_generations):
                prev_best_fitness = best_fitness
                
                # Debug population diversity
                population_fitnesses = [ga_operator.calculate_fitness(ind, self.resource) 
                                      for ind in ga_operator.population.individuals]
                print(f"\nGeneration {generation} Population Stats:")
                print(f"Mean Fitness: {sum(population_fitnesses)/len(population_fitnesses):.2f}")
                print(f"Best Fitness: {min(population_fitnesses):.2f}")
                print(f"Worst Fitness: {max(population_fitnesses):.2f}")
                print(f"Fitness Variance: {np.var(population_fitnesses):.2f}")

                # Selection with debug
                parents = ga_operator.select_mating_pool(ga_operator.population, self.num_parents_mating)
                parent_fitnesses = [ga_operator.calculate_fitness(p, self.resource) 
                                  for p in parents.individuals]
                print("\nSelected Parents Stats:")
                print(f"Mean Parent Fitness: {sum(parent_fitnesses)/len(parent_fitnesses):.2f}")
                print(f"Best Parent: {min(parent_fitnesses):.2f}")
                
                # Crossover & Mutation với debug
                offspring_crossover = ga_operator._one_point_crossover(
                    parents.individuals[0], parents.individuals[1]
                )
                # Convert tuple to list of individuals
                offspring_list = [offspring_crossover[0], offspring_crossover[1]]
                offspring_population = Population()
                offspring_population.individuals = offspring_list

                offspring_mutation = ga_operator.mutation(
                    offspring_population,
                    self.genetic_params.mutation_rate,
                    self.power_supply_capacity_max
                )

                if offspring_mutation.individuals:  # Check if there are any offspring
                    offspring_fitnesses = [ga_operator.calculate_fitness(ind, self.resource) 
                                         for ind in offspring_mutation.individuals]
                    print("\nOffspring Stats:")
                    print(f"Mean Offspring Fitness: {sum(offspring_fitnesses)/len(offspring_fitnesses):.2f}")
                    print(f"Best Offspring: {min(offspring_fitnesses):.2f}")
                else:
                    print("\nNo valid offspring produced in this generation")

                # Selection
                parents = ga_operator.select_mating_pool(
                    ga_operator.population, 
                    self.num_parents_mating
                )
                
                # Crossover
                offspring = ga_operator.crossover(
                    parents, 
                    self.genetic_params.crossover_rate,
                    self.power_supply_capacity_max
                )
                
                # Mutation
                offspring = ga_operator.mutation(
                    offspring, 
                    self.genetic_params.mutation_rate,
                    self.power_supply_capacity_max
                )
                
                # Combine populations and evaluate
                all_individuals = ga_operator.population.individuals + offspring.individuals
                all_fitnesses = [ga_operator.calculate_fitness(ind, self.resource) 
                               for ind in all_individuals]
                
                # Create new population with elitism
                sorted_indices = np.argsort(all_fitnesses)
                new_population = Population()
                
                # Add elites
                for i in range(self.num_elites):
                    new_population.individuals.append(
                        copy.deepcopy(all_individuals[sorted_indices[i]])
                    )
                
                # Fill rest through tournament selection
                while len(new_population.individuals) < self.popsize:
                    tournament = random.sample(all_individuals, self.tournament_size)
                    winner = min(tournament, 
                               key=lambda x: ga_operator.calculate_fitness(x, self.resource))
                    new_population.individuals.append(copy.deepcopy(winner))
                
                # Update population
                ga_operator.population = new_population
                
                # Update best solution
                current_best_fitness = min(all_fitnesses)
                if current_best_fitness < best_fitness:
                    best_fitness = current_best_fitness
                    best_individual = copy.deepcopy(
                        all_individuals[np.argmin(all_fitnesses)]
                    )
                    generations_without_improvement = 0
                else:
                    generations_without_improvement += 1
                
                # Early stopping
                if generations_without_improvement >= 50:
                    print("\nEarly stopping - No improvement for 50 generations")
                    break
                    
                # Logging
                if generation % 1 == 0:
                    print(f"""
                        Generation {generation}:
                        Time Elapsed: {time.time() - start_time:.2f}s
                        Current Best Fitness: {current_best_fitness:.2f}
                        Overall Best Fitness: {best_fitness:.2f}
                        Generations without improvement: {generations_without_improvement}
                    """)
                
                # Save checkpoints
                if generation % 50 == 0:
                    PopulationSerializer.save_population_to_json(
                        ga_operator.population,
                        f"./save_data/population_gen_{generation}.json"
                    )
                    
                fitness_history.append(current_best_fitness)
                
        # Save final results
        save_evolution_charts([], fitness_history, self.num_generations)
        PopulationSerializer.save_population_to_json(
            Population([best_individual]), 
            "./save_data/best_solution.json"
        )
        
        print(f"\nOptimization completed in {time.time() - start_time:.2f} seconds")
        return best_individual, best_fitness