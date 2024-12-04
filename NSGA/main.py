from src.parameters import IntervalParams
from src.nsga.utils import Utils  # Giữ lại vì vẫn cần các utility functions
from src.algorithm import NsgaAlgorithm  # Đổi tên class import
from src.all_packages import PATHS


def main_nsga():  # Đổi tên function
    # Algorithm parameters
    popsize = 50
    # Number of individuals used for crossover and mutation
    num_parents_mating = 30
    # Maximum battery capacity in the resource table
    power_supply_capacity_max = 10
    # Supply limit for a specific session
    session_capacity_threshold = 4
    
    # Bỏ crossover_params và mutation_params vì không cần thiết trong NSGA-II thuần túy
    
    interval_params = IntervalParams(save_pareto=50, save_population=300)
    # Number of iterations of the algorithm
    num_generations = 10
    # Load population has been saved in json file
    is_load_data = False

    nsga_algorithm = NsgaAlgorithm(  # Đổi tên instance
        popsize,
        num_parents_mating,
        power_supply_capacity_max,
        session_capacity_threshold,
        interval_params,  # Bỏ crossover_params và mutation_params
        num_generations,
        is_load_data,
    )
    nsga_algorithm.run_algorithm()
    
if __name__ == "__main__":
    Utils.print_current_time("Start")
    main_nsga()  # Đổi tên function call
    Utils.print_current_time("End")