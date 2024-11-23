from src.parameters import CrossoverParams, IntervalParams, MutationParams
from src.pmsbx_nsga.utils import Utils
from src.algorithm import PmsbxNsgaAlgorithm
from src.all_packages import PATHS


def main_pmsbx_nsga():
    # Algorithm parameters
    popsize = 50
    # Number of individuals used for crossover and mutation
    num_parents_mating = 30
    # Maximum battery capacity in the resource table
    power_supply_capacity_max = 10
    # Supply limit for a specific session
    session_capacity_threshold = 4
    crossover_params = CrossoverParams(
        time_of_day=1, diff_date=5, battery_type=2
    )
    mutation_params = MutationParams(
        time_of_day=0.5, diff_date=10, battery_type=4
    )
    interval_params = IntervalParams(save_pareto=50, save_population=300)
    # Number of iterations of the algorithm
    num_generations = 3000
    # Load population has been saved in json file
    is_load_data = False

    pmsbx_nsga_algorithm = PmsbxNsgaAlgorithm(
        popsize,
        num_parents_mating,
        power_supply_capacity_max,
        session_capacity_threshold,
        crossover_params,
        mutation_params,
        interval_params,
        num_generations,
        is_load_data,
    )
    pmsbx_nsga_algorithm.run_algorithm()
    
if __name__ == "__main__":
    Utils.print_current_time("Start")
    main_pmsbx_nsga()
    Utils.print_current_time("End")
