import csv
import datetime
import logging
import math
from datetime import datetime
import numpy as np
from collections import namedtuple


class Utils:
    def __init__(self):
        pass

    battery_type_dec = {
        "JSU": 1,
        "LEK": 2,
        "NAJ": 3,
        "POW": 4,
        "ZSF": 5,
    }

    battery_type_dec_to_str = {
        1: "JSU",
        2: "LEK",
        3: "NAJ",
        4: "POW",
        5: "ZSF",
    }

    # Định nghĩa namedtuple với tên là Result
    Gen_structure = namedtuple(
        "Gen_structure",
        [
            "id",
            "start_date",
            "end_date",
            "scheduled_date",
            "time_of_day",
            "battery_type",
            "power_supply_capacity",
        ],
    )

    @staticmethod
    def calculate_num_genes_per_chromo(supply_orders, session_capacity_threshold):
        try:
            # Print column names for debugging
            print("Available columns:", supply_orders.columns.tolist())
            
            # Make sure to use the exact column name from your CSV
            max_total_expected = supply_orders["total_expected"].max()
            return math.ceil(max_total_expected / session_capacity_threshold)
        except KeyError as e:
            print(f"Column not found: {e}")
            print("Please check if the column name matches exactly with the CSV file")
            raise

    # Định nghĩa namedtuple với tên là vector để lưu giá trị của vector sau khi tính toán
    # v = (routine,(scheduled_date − start_date), battery_type, num_battery)
    Vector = namedtuple(
        "Vector",
        ["time_of_day", "difference_date", "battery_type", "power_supply_capacity"],
    )

    @staticmethod
    def get_diesel_schedule(scheduled_date, battery_type, resource):
        # Kiểm tra xem ngày và loại pin có trong dữ liệu không
        if scheduled_date in resource and battery_type in resource[scheduled_date]:
            diesel_morning = resource[scheduled_date][battery_type].get("DIESEL_MORNING")
            diesel_afternoon = resource[scheduled_date][battery_type].get("DIESEL_AFTERNOON")
            return diesel_morning, diesel_afternoon
        else:
            return None, None


    @staticmethod
    def load_data(file_path):
        data = {}
        with open(file_path, mode="r") as file:
            reader = csv.DictReader(file)
            for row in reader:
                date = row["DATE"]
                battery = row["BATTERY"]
                diesel_morning = int(row["DIESEL_MORNING"])
                diesel_afternoon = int(row["DIESEL_AFTERNOON"])

                if date not in data:
                    data[date] = {}
                data[date][battery] = {
                    "DIESEL_MORNING": diesel_morning,
                    "DIESEL_AFTERNOON": diesel_afternoon
                }
        return data

    @staticmethod
    def find_min_value(data):
        min_value = float("inf")  # Khởi tạo giá trị nhỏ nhất là vô cực
        for date, batteries in data.items():
            for battery, value in batteries.items():
                if value < min_value:
                    min_value = value
        return min_value

    @staticmethod
    def covert_battery_type_list_to_numbers(battery_type_list: list):
        battery_type_list_covert_to_numbers = []
        for b in battery_type_list:
            battery_type_list_covert_to_numbers.append(Utils.battery_type_dec[b])
        return battery_type_list_covert_to_numbers

    @staticmethod
    def print_current_time(start_end: str):
        current_time = datetime.now()
        formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")

        # Set up logging configuration to log to a file
        logging.basicConfig(
            filename="logfile.log",  # Specify the log file name
            level=logging.INFO,  # Set the logging level
            format="%(asctime)s - %(message)s",  # Format the log entry
            datefmt="%Y-%m-%d %H:%M:%S",  # Date format for the log entry
        )

        # Log the current time
        print(f"{start_end} time: {formatted_time}")
        logging.info(f"{start_end} time: {formatted_time}")
