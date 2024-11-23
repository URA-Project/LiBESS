from src.parameters import GeneticParams
from src.ga.utils import Utils
from src.algorithm import GaAlgorithm
from src.all_packages import PATHS

def main():
    # Algorithm parameters
    popsize = 200
    num_parents_mating = 60
    power_supply_capacity_max = 10
    session_capacity_threshold = 4
    num_generations = 10
    is_load_data = False
    tournament_size = 3
    num_elites = 5
    
    # GA parameters
    genetic_params = GeneticParams(
        crossover_rate=0.8,
        mutation_rate=0.1
    )
    
    # Initialize and run algorithm
    ga_algorithm = GaAlgorithm(
        popsize=popsize,
        num_parents_mating=num_parents_mating,
        power_supply_capacity_max=power_supply_capacity_max,
        session_capacity_threshold=session_capacity_threshold,
        genetic_params=genetic_params,
        num_generations=num_generations,
        is_load_data=is_load_data,
        tournament_size=tournament_size,
        num_elites=num_elites
    )
    
    ga_algorithm.run_algorithm()

if __name__ == "__main__":
    Utils.print_current_time("Start")
    main()
    Utils.print_current_time("End")
