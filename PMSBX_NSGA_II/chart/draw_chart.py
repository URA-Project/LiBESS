import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Dữ liệu pareto font
A = [
   [[68, 0], [55, 0], [66, 0], [63, 11], [67, 0], [64, 14], [71, 0], [60, 0], [64, 0], [61, 0], [72, 0], [71, 12], [70, 0], [69, 0], [65, 0], [58, 0], [53, 0], [59, 0], [73, 0], [69, 12], [68, 12], [65, 5], [65, 10], [63, 8], [70, 10], [66, 14], [67, 12], [68, 14], [62, 9], [63, 7], [69, 5], [66, 15], [65, 7], [69, 11], [64, 28], [61, 15], [66, 19], [61, 26], [60, 19], [65, 12], [63, 20], [56, 26], [64, 19], [65, 17], [65, 20], [67, 13], [64, 12], [57, 13], [71, 7], [59, 15]],
 [[55, 0], [60, 0], [61, 0], [53, 0], [58, 0], [59, 0], [54, 7], [48, 17], [52, 10], [48, 21], [47, 16], [50, 7], [51, 10], [48, 20], [53, 7], [54, 8], [56, 6], [49, 17], [50, 15], [53, 6], [57, 0], [52, 8], [57, 4], [51, 7], [46, 20], [47, 20], [55, 6], [59, 1], [47, 17], [54, 3], [59, 5], [51, 13], [55, 7], [54, 6], [46, 16], [52, 9], [51, 5], [52, 7], [51, 12], [54, 2], [56, 4], [50, 10], [57, 2], [58, 2], [52, 11], [52, 5], [45, 22], [46, 15], [55, 1], [57, 8]],
[[48, 0], [40, 2], [45, 0], [46, 0], [47, 0], [41, 2], [39, 3], [44, 0], [38, 5], [40, 3], [41, 1], [37, 6], [39, 4], [40, 1], [36, 9], [36, 7], [42, 1], [39, 2], [35, 11], [34, 5], [38, 4], [39, 1], [36, 8], [35, 9], [34, 7], [43, 0], [38, 3], [37, 5], [37, 3], [37, 4], [38, 2], [34, 11], [36, 2], [38, 1], [35, 7], [37, 2], [35, 4], [40, 0], [37, 1], [36, 6], [34, 10], [35, 8], [35, 10], [36, 4], [36, 5], [34, 13], [34, 8], [37, 8], [43, 2], [46, 1]]

]

# Tạo file PDF để lưu các chart
with PdfPages('charts.pdf') as pdf:
    for i, data in enumerate(A):
        x_values = [point[0] for point in data]
        y_values = [point[1] for point in data]

        plt.figure(figsize=(6, 4))
        plt.scatter(x_values, y_values, color='blue')
        plt.text(0.95, 0.95, f'Iteration 1000', ha='right', va='top', transform=plt.gca().transAxes, fontsize=14, color='red')
        plt.xlabel('Total number of deadline violations')
        plt.ylabel('Total number of battery type violations')
        plt.tight_layout()
        plt.grid(True)
        pdf.savefig()
        plt.close()
        # plt.legend()
        # plt.show()

print("The charts have been saved successfully.")