from utils import *
import datetime
from collections import defaultdict
import numpy as np


def hard_constraint_1(chromosome):
    HC1_count = 0
    BATDAY = dict()
    for gen in chromosome:
        # gen_parser = "id","start_date","end_date","scheduled_date","routine","battery_type","num_battery"
        gen_parser = parser_gen_pmsbx(gen)
        ###Decode Setup###
        battery_type = access_row_by_wonum(gen_parser.id)["battery_type"]
        battery_type_list = battery_type.split("|")
        check_battery_type = (
            battery_type_dec_convert[gen_parser.battery_type] in battery_type_list
        )
        # Kiểm tra battery_type của gene hiện tại có nằm trong danh sách của battery_type của id đó trong file data không
        if check_battery_type == False:
            HC1_count += 1
            continue
        device = access_row_by_wonum(gen_parser.id)["device_type"]
        tup_temp = (
            battery_type_dec_convert[gen_parser.battery_type],
            gen_parser.scheduled_date,
            device,
        )
        BATDAY[tup_temp] = BATDAY.get(tup_temp, 0) + 0.5

    for key, value in BATDAY.items():
        bat_type, date, device = key
        data_resource_value = get_resource(bat_type, date, device)
        # Chuyen value (đang đơn vị là day) thành ngày theo công thức 1 day = 10KAh
        value_KAh = value * 10
        if data_resource_value == -1:  # gen date with date not in resouce
            HC1_count += 1
        elif data_resource_value < value_KAh:
            HC1_count += 1
    return HC1_count


def hard_constraint_2(chromosome):
    # Ở một thời điểm, mỗi thiết bị chỉ được cung cấp năng lượng bởi 1 pin mà thôi
    # (không có 2 pin cùng cung cấp năng lượng cho 1 máy)
    HC2_count = 0
    parsed_genes = [parser_gen_pmsbx(gene) for gene in chromosome]
    device_types = {
        gene.id: access_row_by_wonum(gene.id)["device_type"] for gene in parsed_genes
    }

    num_genes = len(parsed_genes)

    for i in range(num_genes):
        for j in range(i + 1, num_genes):
            gen_parser = parsed_genes[i]
            temporary_gene_parser = parsed_genes[j]

            if (
                gen_parser.scheduled_date == temporary_gene_parser.scheduled_date
                and gen_parser.routine == temporary_gene_parser.routine
                and device_types[gen_parser.id]
                == device_types[temporary_gene_parser.id]
                and gen_parser.battery_type != temporary_gene_parser.battery_type
            ):
                HC2_count += 1

    return HC2_count


def hard_constraint_3(chromosome):
    # Ở một thời điểm, một pin không cung cấp năng lượng cho 2 thiết bị khác nhau.
    HC3_count = 0
    parsed_genes = [parser_gen_pmsbx(gene) for gene in chromosome]
    device_types = {
        gene.id: access_row_by_wonum(gene.id)["device_type"] for gene in parsed_genes
    }

    num_genes = len(parsed_genes)

    for i in range(num_genes):
        for j in range(i + 1, num_genes):
            gen_parser = parsed_genes[i]
            temporary_gene_parser = parsed_genes[j]

            if (
                gen_parser.scheduled_date == temporary_gene_parser.scheduled_date
                and gen_parser.routine == temporary_gene_parser.routine
                and device_types[gen_parser.id]
                != device_types[temporary_gene_parser.id]
                and gen_parser.battery_type == temporary_gene_parser.battery_type
            ):
                HC3_count += 1

    return HC3_count


def soft_constraint_1(chromosome):
    # The scheduled date of a request s_i should happen closely to
    # the time that the request s_i−1 before it occurs, or f_diff(si, ei) − f_diff(si−1, ei−1) → 0.
    SC_count = 0
    for index, current_gene in enumerate(chromosome):
        if index == 0:
            continue
        # gen_parser = "id","start_date","end_date","scheduled_date","routine","battery_type","num_battery"
        gene_parser = parser_gen_pmsbx(current_gene)
        gene_before_parser = parser_gen_pmsbx(chromosome[index - 1])
        diff_date = difference_date(
            gene_parser.scheduled_date, gene_before_parser.scheduled_date
        )
        date_count = abs(diff_date.days)
        # date_count là khoảng cách của số ngày thứ i và số ngày thứ i-1.
        # Theo yêu cầu soft constraint 1 trong bài báo thì
        # 2 ngày thứ i và số ngày thứ i-1 càng gần nhau càng tốt
        SC_count = date_count + SC_count
    return SC_count


def soft_constraint_2(chromosome):
    # Execution time should be minimized f_diff(si, ei) → 0.
    return 0


def soft_constraint_3(chromosome):
    # Việc cung cấp năng lượng nên được thực hiện liên tục, không ngắt
    # quãng hoặc thời gian (duration) để thực hiện công việc này không
    # nên quá dài (càng ngắn càng tốt)

    # Bước 1. Duyệt gene trong chromosome
    # Bước 2. Tạo mảng và thêm các ngày có cùng Supply_ID vào mảng đó
    # (Mục đích là để so sánh khoảng cách của các ngày thực thi trong cùng Supply_ID)
    # Bước 3. Sắp xếp theo thứ tự ngày tăng dần
    # Bước 4. Tính khoảng cách của các ngày trong cùng ID đó. Nếu khoảng cách lớn hơn 2
    # thì tăng SC_count lên 1

    # Lưu số đếm cho SC, trong trường hợp số ngày của các gene có trong cùng 1 ID mà có khoảng cách lớn hơn 2
    # thì tăng SC_count lên 1
    SC_count = 0
    list_date_same_ID = []
    # 1. Duyệt gene trong chromosome
    for index in range(len(chromosome) - 1):
        # gen_parser = "id","start_date","end_date","scheduled_date","routine","battery_type","num_battery"
        gene_parser = parser_gen_pmsbx(chromosome[index])
        gene_after_parser = parser_gen_pmsbx(chromosome[index + 1])
        if gene_parser.id == gene_after_parser.id:
            scheduled_date = datetime.datetime.strptime(
                gene_parser.scheduled_date, date_format
            )
            list_date_same_ID.append(scheduled_date)
        else:
            sorted_dates = sorted(list_date_same_ID)
            for i in range(len(sorted_dates) - 1):
                diff_date = sorted_dates[i + 1] - sorted_dates[i]
                date_count = abs(diff_date.days)
                if date_count > 2:
                    SC_count = date_count + SC_count
            list_date_same_ID.clear()

    return SC_count


def manday_chromosome(chromosome):
    HC_count = 0
    dealine_count = 0
    result = 0
    BATDAY = defaultdict(float)

    for gen in chromosome:
        gen_parser = parser_gen_pmsbx(gen)
        d_estdur = access_row_by_wonum(gen_parser.id)["d_estdur"]
        device = access_row_by_wonum(gen_parser.id)["device_type"]
        dealine = access_row_by_wonum(gen_parser.id)["end_date"]
        date_dealine = datetime.datetime.strptime(dealine, "%d/%m/%Y")
        date_begin = datetime.datetime.strptime(gen_parser.scheduled_date, "%d/%m/%Y")

        num_date = round(d_estdur)
        check_dealine = date_begin + datetime.timedelta(days=num_date)
        if date_dealine < check_dealine:
            dealine_count += 1
        for i in range(int(d_estdur * 2)):  # Loop over half days
            num_date = i / 2
            run_date = date_begin + datetime.timedelta(days=num_date)
            tup_temp = (
                battery_type_dec_convert[gen_parser.battery_type],
                run_date.strftime("%d/%m/%Y"),
                device,
            )
            BATDAY[tup_temp] += 0.5

    for key, value in BATDAY.items():
        bat_type, date, device = key
        data_resource_value = get_resource(bat_type, date, device)
        # data_resource_value = data_resource_value * 1.25
        value_KAh = value * 5

        if data_resource_value == -1 or data_resource_value < value_KAh:
            HC_count += 1

    result = HC_count + dealine_count * 5

    return result


def cal_fitness_value_full_constraint(population, HC_penalt_point, SC_penalt_point):
    fitness = []
    SC_count = 0

    for index in range(len(population)):
        HC_count = 0
        SC_count = 0
        chromosome = population[index]
        HC_count_1 = hard_constraint_1(chromosome)
        HC_count_2 = hard_constraint_2(chromosome)
        HC_count_3 = hard_constraint_3(chromosome)
        HC_count = HC_count_1 + HC_count_2 + HC_count_3

        SC_count_1 = soft_constraint_1(chromosome)
        SC_count_2 = soft_constraint_2(chromosome)
        SC_count_3 = soft_constraint_3(chromosome)
        SC_count = SC_count_1 + SC_count_2 + SC_count_3
        fitness_value = 1 / (
            HC_count * HC_penalt_point + SC_count * SC_penalt_point + 1
        )
        fitness.append(fitness_value)

    fitness = np.asarray(fitness)
    return fitness


def cal_fitness_value(population, HC_penalt_point, SC_penalt_point):
    fitness = np.empty(len(population), dtype=float)
    for index, chromosome in enumerate(population):
        HC_count = manday_chromosome(chromosome)
        fitness_value = 1 / (HC_count + 1)
        fitness[index] = fitness_value

    return fitness
