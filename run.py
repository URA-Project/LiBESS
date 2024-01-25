from all_packages import *
from src.problem import Problem
from src.utils import load_data

if __name__ == "__main__":

    ####################
    ## Đọc data file
    ####################
    df = load_data(PATHS["workoder"])

    ####################
    ## Chạy tìm lịch tối ưu
    ####################
    problem = Problem(df)

    best_schedule = problem.start()
