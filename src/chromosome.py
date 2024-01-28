import numpy as np
import random
from utils import *

"""Create chromosome - GA"""


class CHROMOSOME_GA:
    def __init__(self, df):
        self.HC_resource = []
        self.HC_time = []
        self.df = df
        self.chromosome = self._generate_parent()

    # Generate random date

    def _generate_parent(self):
        genes = []
        for supply_id, start_day, end_date, battery_type in zip(
            self.df.supply_id,
            self.df.start_day,
            self.df.end_date,
            self.df.battery_type,
        ):
            rand_date = random_date_bit(start_day, end_date, random.random())
            routine = random.choice([0, 1])
            battery_type = battery_type.split("|")
            battery_type_gen = random.choice(battery_type)
            battery_type_gen = battery_type_bit[battery_type_gen]

            random_num_battery = random.randint(0, 10)
            # Chuyển số nguyên sang số nhị phân
            num_battery = format(random_num_battery, "04b")
            bitstring = "".join(
                [rand_date, str(routine), battery_type_gen, num_battery]
            )
            chromosome = "-".join([supply_id, start_day, end_date, bitstring])
            genes.append(chromosome)
        return np.asarray(genes)


"""Create chromosome - PMSBX-GA"""


class CHROMOSOME_PMSBX_GA:
    def __init__(self, df):
        self.HC_resource = []
        self.HC_time = []
        self.df = df
        self.chromosome = self._generate_parent()

    # Generate random date

    def _generate_parent(self):
        genes = []
        for supply_id, start_day, end_date, battery_type in zip(
            self.df.supply_id,
            self.df.start_day,
            self.df.end_date,
            self.df.battery_type,
        ):
            rand_date = random_datetime(start_day, end_date)
            routine = random.choice([0, 1])
            battery_type = battery_type.split("|")
            battery_type_gen = random.choice(battery_type)
            battery_type_gen = battery_type_dec[battery_type_gen]

            num_battery = random.randint(0, 10)
            decstring = "-".join(
                [rand_date, str(routine), str(battery_type_gen), str(num_battery)]
            )
            chromosome = "-".join([supply_id, start_day, end_date, decstring])
            genes.append(chromosome)
        return np.asarray(genes)
