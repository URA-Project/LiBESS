import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

def save_evolution_charts(all_generations_data, fitness_history, current_gen=1000):
    with PdfPages('evolution_charts.pdf') as pdf:
        # 1. Vẽ Pareto Front Evolution
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        
        # Chọn 4-5 generations để vẽ
        num_plots = min(5, len(all_generations_data))
        colors = plt.cm.viridis(np.linspace(0, 1, num_plots))
        
        for idx in range(num_plots):
            gen, data = all_generations_data[idx]
            x_values = [point[0] for point in data]
            y_values = [point[1] for point in data]
            
            plt.scatter(x_values, y_values, 
                       color=colors[idx], 
                       alpha=0.6,
                       s=30,
                       label=f'Gen {gen}')
        
        plt.xlabel('Deadline Violations')
        plt.ylabel('Battery Type Violations')
        plt.title('Pareto Front Evolution')
        plt.legend(fontsize='small')
        plt.grid(True)
        
        # 2. Vẽ Fitness History
        plt.subplot(1, 2, 2)
        generations = range(len(fitness_history))
        plt.plot(generations, fitness_history, 'b-', linewidth=1)
        plt.xlabel('Generation')
        plt.ylabel('Best Fitness')
        plt.title('Fitness Evolution')
        plt.grid(True)
        
        plt.tight_layout()
        pdf.savefig()
        plt.close()

    print("Evolution charts have been saved successfully.")

if __name__ == "__main__":
    # Ví dụ sử dụng:
    population_data = A[0]  # Sử dụng dữ liệu pareto front hiện có
    save_evolution_charts(population_data)