import matplotlib.pyplot as plt
import numpy as np
def draw_loss(data):
    x_data = [str(i+1) for i in range(len(data))]

    y_data = data
    plt.plot(x_data, y_data, color='red', linewidth=2.0)
    plt.show()

def draw_two(data, data2):
    x_data = [i for i in range(len(data))]
    x_data2 = [i for i in range(len(data2))]
    y_data = data
    y_data2 = data2
    plt.plot(x_data, y_data, color='red', linewidth=2.0)
    plt.plot(x_data2, y_data2, color='blue', linewidth=2.0)
    plt.show()
