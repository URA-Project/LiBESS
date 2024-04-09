from src.algorithm import GA_Algorithm
from src.all_packages import *
from src.utilities.utils import load_data

if __name__ == "__main__":
    ####################
    ## Đọc data file
    ####################
    df = load_data(PATHS["data"])

    ####################
    ## Chạy tìm lịch tối ưu
    ####################
    # Example usage:
    sol_per_pop = 40
    num_parents_mating = 16
    num_generations = 3
    mutation_rate = 0.3
    HC_penalt_point = 0
    SC_penalt_point = 0

    genetic_algorithm = GA_Algorithm(
        sol_per_pop, num_parents_mating, num_generations, mutation_rate, HC_penalt_point, SC_penalt_point
    )
    genetic_algorithm.run_algorithm()
