import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

root = tk.Tk()

try:
    breath_file = open('experimental.txt')
    breath_matrix = [line.split("	") for line in breath_file]
    breath_points = []
    x = [i for i in range(0, 450, 1)]
    min_value_index = 0
    max_value_index = 0
    min_value = 1
    max_value = 0
    for i in range(len(breath_matrix)):
        summa = 0
        for j in breath_matrix[i]:
            summa += float(j)
        average = summa/208
        if average < min_value:
            min_value = average
            min_value_index = i
        if average > max_value:
            max_value = average
            max_value_index = i
        breath_points.append(average)
    # plt.scatter(x, breath_points)
    # plt.show()
    print('Индекс измерения с самым глубоким выдохом(момент времени tref) ', min_value_index)
    print('Индекс измерения с самым глубоким вдохом ', max_value_index)
except Exception:
    print('Error')
finally:
    breath_file.close()

figure3 = plt.Figure(figsize=(5, 4), dpi=100)
ax3 = figure3.add_subplot(111)
ax3.scatter(x, breath_points, color='b')
scatter3 = FigureCanvasTkAgg(figure3, root)
scatter3.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH)
ax3.legend(['index_price'])
ax3.set_xlabel('Interest Rate')
ax3.set_title('Breating chart')

root.mainloop()