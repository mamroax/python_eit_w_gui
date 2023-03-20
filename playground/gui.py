import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import functions as f


############################################################################################################33333##3#3#
root = tk.Tk()

x, breath_points, ind_max, ind_min = f.breath('experimental.txt', 1)
figure3 = plt.Figure(figsize=(5, 4), dpi=100)
ax3 = figure3.add_subplot(111)
# ax3.scatter(x, breath_points, color='b')
ax3.plot(x, breath_points, color='b')
scatter3 = FigureCanvasTkAgg(figure3, root)
scatter3.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH)
ax3.legend(['average voltage difference'])
ax3.set_xlabel('measurement points')
ax3.set_ylabel('voltage(V)')
ax3.set_title('Breating chart')

x, breath_points, ind_max, ind_min = f.breath('experimental.txt', 3)

figure3 = plt.Figure(figsize=(5, 4), dpi=100)
ax3 = figure3.add_subplot(111)
ax3.plot(x, breath_points, color='b', lw = 2)
scatter3 = FigureCanvasTkAgg(figure3, root)
scatter3.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH)
ax3.legend(['average voltage difference'])
ax3.set_xlabel('measurement points')
ax3.set_ylabel('voltage(V)')
ax3.set_title('Breating chart')

x, breath_points, ind_max, ind_min = f.breath('experimental.txt', 5)
figure3 = plt.Figure(figsize=(5, 4), dpi=100)
ax3 = figure3.add_subplot(111)
ax3.plot(x, breath_points, color='b', lw = 2)
scatter3 = FigureCanvasTkAgg(figure3, root)
scatter3.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH)
ax3.legend(['average voltage difference'])
ax3.set_xlabel('measurement points')
ax3.set_ylabel('voltage(V)')
ax3.set_title('Breating chart')

root.mainloop()