from all_packages import *

battery_type_bit = {
    "JSU": "000",
    "LEK": "001",
    "NAJ": "010",
    "POW": "011",
    "ZSF": "100",
}

device = {
    # add device...
}

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
