class MutationParams:
    """Distribution index of mutation"""

    def __init__(
        self, time_of_day=1, diff_date=15, battery_type=4
    ):
        self.time_of_day = time_of_day
        self.diff_date = diff_date
        self.battery_type = battery_type

class CrossoverParams:
    """Distribution index of crossover"""

    def __init__(
        self, time_of_day=1, diff_date=1, battery_type=1
    ):
        self.time_of_day = time_of_day
        self.diff_date = diff_date
        self.battery_type = battery_type

class IntervalParams:
    """Interval between tasks: save pareto, save population"""

    def __init__(self, save_pareto=50, save_population=100):
        self.save_pareto = save_pareto
        self.save_population = save_population