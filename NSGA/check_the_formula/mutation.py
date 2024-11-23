import numpy as np


def date_function():
    random_rate = 0.01
    distribution_index_values = [5, 10, 15]  # Nên là 10
    random_date = [1, 5, 10, 15, 20, 25, 30]
    date_array = [1, 5, 10, 15, 20, 25, 30]
    new_date = 0

    for date in date_array:
        print(f"**************** Date: {date} ****************************")
        for distribution_index in distribution_index_values:
            if random_rate <= 0.5:
                delta = (2 * random_rate) ** (1 / (distribution_index + 1)) - 1
            else:
                delta = 1 - (2 * (1 - random_rate)) ** (1 / (distribution_index + 1))
            if random_rate <= 0.5:
                for n in random_date:
                    new_date = date + delta * (date - n)
                    print(
                        f"Random rate < 0.5 **** Distribution_index: {distribution_index} *** Random date: {n} *** Delta: {delta} **** New_date: {new_date}"
                    )
            else:
                for n in random_date:
                    new_date = date + delta * (n - date)
                    print(
                        f"Random rate > 0.5 **** Distribution_index: {distribution_index} *** Random date: {n} *** Delta: {delta} **** New_date: {new_date}"
                    )


def battery_type_function():
    random_rate = 0.8
    distribution_index_values = [1, 2, 3, 4, 5]  # Nên là 2
    # distribution_index_values = [2]
    random_battery_type = [1, 2, 3, 4, 5]
    # random_battery_type = [2]
    battery_type_array = [1, 2, 3, 4, 5]
    # battery_type_array = [3]
    new_battery_type = 0

    for battery_type in battery_type_array:
        print(
            f"**************** Battery type: {battery_type} ****************************"
        )
        for distribution_index in distribution_index_values:
            if random_rate <= 0.5:
                delta = (2 * random_rate) ** (1 / (distribution_index + 1)) - 1
            else:
                delta = 1 - (2 * (1 - random_rate)) ** (1 / (distribution_index + 1))
            if random_rate <= 0.5:
                for n in random_battery_type:
                    new_battery_type = battery_type + delta * (battery_type - n)
                    print(
                        f"ε: {random_rate} **** η: {distribution_index} *** a: {n} *** Delta: {delta} *** Battery_type: {battery_type} *** New battery type: {new_battery_type}"
                    )
            else:
                for n in random_battery_type:
                    new_battery_type = battery_type + delta * (n - battery_type)
                    print(
                        f"ε: {random_rate} **** η: {distribution_index} *** a: {n} *** Delta: {delta} *** Battery_type: {battery_type} *** New battery type: {new_battery_type}"
                    )


def power_supply_capacity_mutation():
    random_rate = 0.9
    distribution_index_values = [5, 10]  # Nên là 5
    random_battery_type = [1, 3, 5, 7, 9]
    battery_type_array = [1, 2, 4, 6, 8, 9]
    new_battery_type = 0

    for battery_type in battery_type_array:
        print(
            f"**************** Power_supply_capacity: {battery_type} ****************************"
        )
        for distribution_index in distribution_index_values:
            if random_rate <= 0.5:
                delta = (2 * random_rate) ** (1 / (distribution_index + 1)) - 1
            else:
                delta = 1 - (2 * (1 - random_rate)) ** (1 / (distribution_index + 1))
            if random_rate <= 0.5:
                for n in random_battery_type:
                    new_battery_type = battery_type + delta * (battery_type - n)
                    print(
                        f"Random rate < 0.5 **** Distribution_index: {distribution_index} *** Random battery type: {n} *** Delta: {delta} **** New Power_supply_capacity: {new_battery_type}"
                    )
            else:
                for n in random_battery_type:
                    new_battery_type = battery_type + delta * (n - battery_type)
                    print(
                        f"Random rate > 0.5 **** Distribution_index: {distribution_index} *** Random date: {n} *** Delta: {delta} **** New Power_supply_capacity: {new_battery_type}"
                    )


def time_of_date_mutation():
    random_rate = 0.9
    distribution_index_values = [0.5, 1]  # Nên là 5
    random_battery_type = [0, 1]
    battery_type_array = [0, 1]
    new_battery_type = 0

    for battery_type in battery_type_array:
        print(
            f"**************** Power_supply_capacity: {battery_type} ****************************"
        )
        for distribution_index in distribution_index_values:
            if random_rate <= 0.5:
                delta = (2 * random_rate) ** (1 / (distribution_index + 1)) - 1
            else:
                delta = 1 - (2 * (1 - random_rate)) ** (1 / (distribution_index + 1))
            if random_rate <= 0.5:
                for n in random_battery_type:
                    new_battery_type = battery_type + delta * (battery_type - n)
                    print(
                        f"Random rate < 0.5 **** Distribution_index: {distribution_index} *** Random battery type: {n} *** Delta: {delta} **** New Power_supply_capacity: {new_battery_type}"
                    )
            else:
                for n in random_battery_type:
                    new_battery_type = battery_type + delta * (n - battery_type)
                    print(
                        f"Random rate > 0.5 **** Distribution_index: {distribution_index} *** Random date: {n} *** Delta: {delta} **** New Power_supply_capacity: {new_battery_type}"
                    )


if __name__ == "__main__":
    time_of_date_mutation()
