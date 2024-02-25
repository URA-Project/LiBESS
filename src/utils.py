import datetime
import random
from all_packages import *
import numpy as np
from collections import namedtuple

battery_type_bit = {
    "JSU": "000",
    "LEK": "001",
    "NAJ": "010",
    "POW": "011",
    "ZSF": "100",
}

battery_type_dec = {
    "JSU": "0",
    "LEK": "1",
    "NAJ": "2",
    "POW": "3",
    "ZSF": "4",
}

device = {
    # add device...
}

date_format = "%d/%m/%Y"

# Global data
global data_frame


def load_data(path) -> pd.DataFrame:
    assert osp.isfile(path)
    df = pd.read_csv(path)
    return df


def pre_process_data():
    df = load_data(PATHS["data"])
    df = df.drop("Unnamed: 0", axis=1)
    df = df[df.battery_type != "Not Defined"]
    df = df.dropna().reset_index(drop=True)
    return df


data_frame = pre_process_data()


# Create a dictionary in Python to map between the supply_id values and the indices of the DataFrame data.
# Key (x): supply_id value
# Value (y): corresponding index in the DataFrame data
def get_dict_supply_id():
    return {x: y for x, y in zip(data_frame.supply_id, data_frame.index)}


# Access a row in the DataFrame data based on the supply_id value using the dict_supply_id dictionary.
def access_row_by_wonum(supply_id):
    dict_supply_id = get_dict_supply_id()
    return data_frame.iloc[dict_supply_id[supply_id]]


def get_resource(battery_type, date, capacity_ah):
    resource_data = load_data(PATHS["resource"])
    date_unique = np.unique(resource_data.date.to_list()).astype(list)
    if date not in date_unique:
        return -1
    return resource_data[
        (resource_data["battery_type"] == battery_type)
        & (resource_data["date"] == date)
    ][capacity_ah].item()


def get_data():
    return data_frame


def _str_time_prop(start, end, time_format, prop):
    stime = datetime.datetime.strptime(start, time_format)
    etime = datetime.datetime.strptime(end, time_format)
    ptime = stime + prop * (etime - stime)
    ptime = ptime.strftime("%d/%m/%Y")
    return ptime


# Random date format bits
def random_date_bit(start, end, prop):  # 0001 = current year, 0002 = next year
    # generate date in current data
    sched_start = _str_time_prop(start, end, "%d/%m/%Y", prop)
    date_sched_start = format(int(sched_start[:2]), "05b")
    month_sched_start = format(int(sched_start[3:5]), "04b")
    year_sched_start = format(int(sched_start[6:]), "02b")
    sched_start = "".join([date_sched_start, month_sched_start, year_sched_start])
    return sched_start


# Random date by method creates a datetime object from the given string
def random_datetime(date_begin, date_end):
    # Chuyển đổi ngày tháng bắt đầu và kết thúc thành đối tượng datetime
    dt_begin = datetime.datetime.strptime(date_begin, date_format)
    dt_end = datetime.datetime.strptime(date_end, date_format)

    # Tính số ngày giữa dt_begin và dt_end
    delta = dt_end - dt_begin

    # Sinh ngẫu nhiên số ngày và thêm vào dt_begin
    random_days = random.randint(0, delta.days)
    result_datetime = dt_begin + datetime.timedelta(days=random_days)

    # Định dạng thành chuỗi ngày tháng
    result_datetime_string = result_datetime.strftime("%d/%m/%Y")

    return result_datetime_string

# Sử dụng hàm để lấy số ngẫu nhiên từ 0 đến 1
def generate_random_number():
    return np.random.rand()

def power_of_fraction(base, numerator, denominator):
    # Ví dụ: tính 2^(1/3)
    # base = 2
    # numerator = 1
    # denominator = 3
    result = np.power(base, numerator / denominator)
    return result

# Định nghĩa namedtuple với tên là Result
Gen_structure = namedtuple('Gen_structure', ['id', 'start_date', 'end_date', 'scheduled_date', 'routine', 'battery_type', 'num_battery'])

def parser_gen_pmsbx(gen):
    # Tách chuỗi bởi dấu "-"
    result_list = gen.split('-')

    # Gán giá trị tách ra từng biến riêng biệt
    id = result_list[0]
    start_date = result_list[1]
    end_date = result_list[2]
    scheduled_date = result_list[3]
    routine = result_list[4]
    battery_type = result_list[5]
    num_battery = result_list[6]
    # Trả về một đối tượng Result với các giá trị tương ứng
    return Gen_structure(id, start_date, end_date, scheduled_date, routine, battery_type, num_battery)

# Định nghĩa namedtuple với tên là vector để lưu giá trị của vector sau khi tính toán
# v = (routine,(scheduled_date − start_date), battery_type, num_battery)
Vector = namedtuple('Vector', ['routine', 'difference_date', 'battery_type', 'num_battery'])

def scalar_multiply_v1_crossover(beta, vector_u1, vector_u2):
    #v1_new = 0.5 × [(1 + β)υ1 + (1 − β)υ2]
    vector_v1 = tuple(0.5 * ( (1 + beta) * u1 + (1 - beta) * u2) for u1, u2 in zip(vector_u1, vector_u2))
    rounded_vector = tuple(int(round(element)) for element in vector_v1)
    return Vector(*rounded_vector)

def scalar_multiply_v2_crossover(beta, vector_u1, vector_u2):
    #v2_new = 0.5 × [(1 - β)υ1 + (1 + β)υ2]
    vector_v2 = tuple(0.5 * ( (1 - beta) * u1 + (1 + beta) * u2) for u1, u2 in zip(vector_u1, vector_u2))
    rounded_vector = tuple(int(round(element)) for element in vector_v2)
    return Vector(*rounded_vector)

def difference_date(date_1, date_2):
    # date_1, date_2: string
    scheduled_date = datetime.datetime.strptime(date_1, date_format)
    start_date = datetime.datetime.strptime(date_2, date_format)
    difference = scheduled_date - start_date
    return difference

def scalar_multiply_motation(vector, delta, epsilon ):
    new_vector = (0,0,0,0)
    # ['routine', 'difference_date', 'battery_type', 'num_battery']
    random_routine = random.randint(0,1)
    random_diff_date = random.randint(0,30)
    random_battery_type = random.randint(1,5)
    random_num_battery = random.randint(1,10)
    vector_random = (random_routine,random_diff_date,random_battery_type,random_num_battery)
    if epsilon <= 0.5 :
        #v_new = v + δ × [υ − (a1, a2, a3, a4)] for ε ≤ 0.5
        new_vector = tuple( u1 + delta * (u1 - u2) for u1, u2 in zip(vector, vector_random))
    else :
        #v_new = v + δ × [(a1, a2, a3, a4) - υ] for ε ≤ 0.5
        new_vector = tuple( u1 + delta * (u2 - u1) for u1, u2 in zip(vector, vector_random))
    rounded_vector = tuple(int(round(element)) for element in new_vector)
    return Vector(*rounded_vector)