import numpy as np


def date_function():
    random_rate = 0.9
    distribution_index_values = [0.5, 1, 2, 5]  # Nên là 1
    v1 = 1
    v2 = 0

    beta = 0

    # for date in date_array:
    # print(f"**************** Date: {date} ****************************")
    for distribution_index in distribution_index_values:
        if random_rate <= 0.5:
            beta = (2 * random_rate) ** (1 / (distribution_index + 1))
            # print(f"distribution_index_values: {distribution_index}, beta: {beta}")
        else:
            beta = (1 / (2 - 2 * random_rate)) ** (1 / (distribution_index + 1))
            # print(f"distribution_index_values: {distribution_index}, beta: {beta}")

        v1_new = 0.5 * ((1 + beta) * v1 + (1 - beta) * v2)
        print(
            f"Random rate < 0.5 **** Distribution_index: {distribution_index} *** Beta: {beta} *** V1: {v1} *==>: {v1_new}"
        )

        v2_new = 0.5 * ((1 - beta) * v1 + (1 + beta) * v2)
        print(
            f"Random rate < 0.5 **** Distribution_index: {distribution_index} *** Beta: {beta} *** V2: {v2} *==>: {v2_new}"
        )


if __name__ == "__main__":
    date_function()
