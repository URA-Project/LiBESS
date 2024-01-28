import datetime
import numpy as np
import random
from utils import *


def _str_time_prop(start, end, time_format, prop):
    stime = datetime.datetime.strptime(start, time_format)
    etime = datetime.datetime.strptime(end, time_format)
    ptime = stime + prop * (etime - stime)
    ptime = ptime.strftime("%d/%m/%Y")
    return ptime


def random_date(start, end, prop):  # 0001 = current year, 0002 = next year
    # generate date in current data
    sched_start = _str_time_prop(start, end, "%d/%m/%Y", prop)
    date_sched_start = format(int(sched_start[:2]), "05b")
    month_sched_start = format(int(sched_start[3:5]), "04b")
    year_sched_start = format(int(sched_start[6:]), "02b")
    sched_start = "".join([date_sched_start, month_sched_start, year_sched_start])
    return sched_start


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
            rand_date = random_date(start_day, end_date, random.random())
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
    pass
