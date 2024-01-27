from algorithm import GA_Algorithm
from all_packages import *
from src.utils import load_data

if __name__ == "__main__":
    ####################
    ## Đọc data file
    ####################
    df = load_data(PATHS["workoder"])

    ####################
    ## Chạy tìm lịch tối ưu
    ####################
    # Example usage:
    sol_per_pop = 40
    num_parents_mating = 16
    num_generations = 3
    mutation_rate = 0.3

    genetic_algorithm = GA_Algorithm(
        sol_per_pop, num_parents_mating, num_generations, mutation_rate
    )
    genetic_algorithm.run_algorithm()
