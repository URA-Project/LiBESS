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
