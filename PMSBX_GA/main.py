from src.parameters import CrossoverParams, MutationParams
from src.pmsbx_ga.utils import Utils
from src.algorithm import PmsbxGaAlgorithm
from src.all_packages import PATHS

def main_pmsbx_ga():
    # Algorithm parameters
    popsize = 200
    num_parents_mating = 60
    power_supply_capacity_max = 10
    session_capacity_threshold = 4
    
    # Crossover parameters
    crossover_params = CrossoverParams(
        time_of_day=2,
        diff_date=10,
        battery_type=3
    )
    
    # Mutation parameters
    mutation_params = MutationParams(
        time_of_day=1,
        diff_date=15,
        battery_type=5
    )
    tournament_size = 7
    num_generations = 10
    is_load_data = False

    # Initialize and run algorithm
    ga_algorithm = PmsbxGaAlgorithm(
        popsize,
        num_parents_mating,
        power_supply_capacity_max,
        session_capacity_threshold,
        crossover_params,
        mutation_params,
        num_generations,
        is_load_data,
        tournament_size=tournament_size
    )
    
    ga_algorithm.run_algorithm()
    
if __name__ == "__main__":
    Utils.print_current_time("Start")
    main_pmsbx_ga()
    Utils.print_current_time("End")
