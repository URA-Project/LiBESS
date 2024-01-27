""" Trong file này mình sẽ định nghĩa các hàm điều kiện, output của chúng sẽ được gộp vào fitness value
"""


def cond_time() -> float:
    pass


def cond_worker() -> float:
    pass


def cond_1() -> float:
    pass


def cond_2() -> float:
    pass


conditions = {"time": cond_time, "worker": cond_worker}


def cal_fitness_value(chromosome) -> float:
    final_fitness_value = 0

    for name, condition in conditions.items():
        final_fitness_value += condition(chromosome)

    return final_fitness_value
