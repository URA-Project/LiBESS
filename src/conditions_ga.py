from utils import*

def hard_constraint_1(chromosome):
    HC1_count = 0
    return HC1_count


def hard_constraint_2(chromosome):
    #Ở một thời điểm, mỗi thiết bị chỉ được cung cấp năng lượng bởi 1 pin mà thôi
    #(không có 2 pin cùng cung cấp năng lượng cho 1 máy)
    HC2_count = 0
    return HC2_count


def hard_constraint_3(chromosome):
    # Ở một thời điểm, một pin không cung cấp năng lượng cho 2 thiết bị khác nhau.
    HC3_count = 0
    return HC3_count

def soft_constraint_1(chromosome):
    # The scheduled date of a request s_i should happen closely to
    # the time that the request s_i−1 before it occurs, or f_diff(si, ei) − f_diff(si−1, ei−1) → 0.
    SC_count = 0
    return SC_count

def soft_constraint_2(chromosome):
    #Execution time should be minimized f_diff(si, ei) → 0.
    return 0

def soft_constraint_3(chromosome):
    #Việc cung cấp năng lượng nên được thực hiện liên tục, không ngắt
    #quãng hoặc thời gian (duration) để thực hiện công việc này không
    #nên quá dài (càng ngắn càng tốt)
    
    #Bước 1. Duyệt gene trong chromosome 
    #Bước 2. Tạo mảng và thêm các ngày có cùng Supply_ID vào mảng đó 
    # (Mục đích là để so sánh khoảng cách của các ngày thực thi trong cùng Supply_ID)
    #Bước 3. Sắp xếp theo thứ tự ngày tăng dần
    #Bước 4. Tính khoảng cách của các ngày trong cùng ID đó. Nếu khoảng cách lớn hơn 2
    # thì tăng SC_count lên 1

    #Lưu số đếm cho SC, trong trường hợp số ngày của các gene có trong cùng 1 ID mà có khoảng cách lớn hơn 2
    # thì tăng SC_count lên 1
    SC_count = 0
    return SC_count


def cal_fitness_value(population, HC_penalt_point, SC_penalt_point):
    fitness = []
    HC_count_print = []
    SC_count_print = []
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
        fitness_value = 1/(HC_count*HC_penalt_point + SC_count*SC_penalt_point + 1)
        fitness.append(fitness_value)
        HC_count_print.append(HC_count)
        SC_count_print.append(SC_count)
    print("HC_count_print", HC_count_print)
    print("SC_count_print",SC_count_print)
    fitness = np.asarray(fitness)
    return fitness