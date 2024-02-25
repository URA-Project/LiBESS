from utils import*


def cond_time() -> float:
    pass


def cond_worker() -> float:
    pass

def hard_constraint_1(chromosome):
    pass

def hard_constraint_2(chromosome):
    #Ở một thời điểm, mỗi thiết bị chỉ được cung cấp năng lượng bởi 1 pin mà thôi 
    #(không có 2 pin cùng cung cấp năng lượng cho 1 máy)
    HC2_count = 0
    parsed_genes = [parser_gen_pmsbx(gene) for gene in chromosome]
    device_types = {gene.id: access_row_by_wonum(gene.id)["device_type"] for gene in parsed_genes}

    num_genes = len(parsed_genes)

    for i in range(num_genes):
        for j in range(i + 1, num_genes):
            gen_parser = parsed_genes[i]
            temporary_gene_parser = parsed_genes[j]

            if (
                gen_parser.scheduled_date == temporary_gene_parser.scheduled_date
                and device_types[gen_parser.id] == device_types[temporary_gene_parser.id]
                and gen_parser.battery_type != temporary_gene_parser.battery_type
            ):
                HC2_count += 1

    return HC2_count


def hard_constraint_3(chromosome):
    HC3_count = 0
    parsed_genes = [parser_gen_pmsbx(gene) for gene in chromosome]
    device_types = {gene.id: access_row_by_wonum(gene.id)["device_type"] for gene in parsed_genes}

    num_genes = len(parsed_genes)

    for i in range(num_genes):
        for j in range(i + 1, num_genes):
            gen_parser = parsed_genes[i]
            temporary_gene_parser = parsed_genes[j]

            if (
                gen_parser.scheduled_date == temporary_gene_parser.scheduled_date
                and device_types[gen_parser.id] != device_types[temporary_gene_parser.id]
                and gen_parser.battery_type == temporary_gene_parser.battery_type
            ):
                HC3_count += 1

    return HC3_count

def soft_constraint_1(chromosome):
    # The scheduled date of a request s_i should happen closely to
    # the time that the request s_i−1 before it occurs, or f_diff(si, ei) − f_diff(si−1, ei−1) → 0.
    SC_count = 0
    for index, current_gene in enumerate(chromosome):
        if(index == 0): continue
        # gen_parser = "id","start_date","end_date","scheduled_date","routine","battery_type","num_battery"
        gene_parser = parser_gen_pmsbx(current_gene)
        gene_before_parser = parser_gen_pmsbx(chromosome[index - 1])
        diff_date = difference_date(gene_parser.scheduled_date, gene_before_parser.scheduled_date)
        date_count = abs(diff_date.days)
        # date_count là khoảng cách của số ngày thứ i và số ngày thứ i-1. Theo yêu cầu soft constraint 1 trong bài báo thì
        # 2 ngày thứ i và số ngày thứ i-1 càng gần nhau càng tốt
        SC_count = date_count + SC_count
    return SC_count

def soft_constraint_2(chromosome):
    return 0

def soft_constraint_3(chromosome):
    return 0

conditions = {"time": cond_time, "worker": cond_worker}


def cal_fitness_value(population, HC_penalt_point, SC_penalt_point):
    final_fitness_value = 0
    HC_count = 0
    SC_count = 0

    for index in range(len(population)):
        chromosome = population[index]
        HC_count_1 = hard_constraint_1(chromosome.chromosome)
        HC_count_2 = hard_constraint_2(chromosome.chromosome)
        HC_count_3 = hard_constraint_3(chromosome.chromosome)
        HC_count = HC_count_1 + HC_count_2 + HC_count_3 + HC_count

        SC_count_1 = soft_constraint_1(chromosome.chromosome)
        SC_count_2 = soft_constraint_2(chromosome.chromosome)
        SC_count_3 = soft_constraint_3(chromosome.chromosome)
        SC_count = SC_count_1 + SC_count_2 + SC_count_3 + SC_count
        
    final_fitness_value = HC_count*HC_penalt_point + SC_count*SC_penalt_point
    return final_fitness_value