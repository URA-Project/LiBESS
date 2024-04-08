import matplotlib.pyplot


def draw_fitness(best_outputs):
    matplotlib.pyplot.plot(best_outputs)
    matplotlib.pyplot.xlabel("Iteration")
    matplotlib.pyplot.ylabel("Fitness")
    matplotlib.pyplot.show()
