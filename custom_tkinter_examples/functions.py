import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
import customtkinter


def breath(file_path, num_of_frames):
    try:
        breath_file = open(file_path)
        breath_matrix = [line.split("	") for line in breath_file]
        breath_points = []
        x = [i for i in range(0, 450, num_of_frames)]
        min_value_index = 0
        max_value_index = 0
        min_value = 1
        max_value = 0
        for i in range(0, len(breath_matrix), num_of_frames):
            summa = 0
            for k in range(num_of_frames):
                for j in breath_matrix[i+k]:
                    summa += float(j)
            average = summa/208/num_of_frames
            if average < min_value:
                min_value = average
                min_value_index = i
            if average > max_value:
                max_value = average
                max_value_index = i
            breath_points.append(average)
    except Exception:
        print('Error')
    finally:
        breath_file.close()
    return [x, breath_points, min_value_index, max_value_index]

def build_graph(root: tk.Tk, path, number_of_frames):
    x, breath_points, ind_max, ind_min = breath('experimental.txt', number_of_frames)
    figure3 = plt.Figure(figsize=(5, 4), dpi=100)
    ax3 = figure3.add_subplot(111)
    # ax3.scatter(x, breath_points, color='b')
    ax3.plot(x, breath_points, color='b')
    scatter3 = FigureCanvasTkAgg(figure3, root)
    scatter3.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH)
    ax3.legend(['average voltage difference'])
    ax3.set_xlabel('measurement points')
    ax3.set_ylabel('voltage(V)')
    ax3.set_title('Breating chart')


class Table:
    def __init__(self, root:tk.Tk, lst:list[set]):
        # code for creating table
        total_rows = len(lst)
        total_columns = len(lst[0])
        for i in range(total_rows):
            for j in range(total_columns):
                self.e = tk.Entry(root, width=20, fg='blue',
                               font=('Arial', 16, 'bold'))

                self.e.grid(row=i, column=j)
                self.e.insert(tk.END, lst[i][j])


def make_table(root: tk.Tk):
    # take the data
    list1 = [(1, 'Raj', 'Mumbai', 19),
             (2, 'Aaryan', 'Pune', 18),
             (3, 'Vaishnavi', 'Mumbai', 20),
             (4, 'Rachna', 'Mumbai', 21),
             (5, 'Shubham', 'Delhi', 21)]
    # create root window
    # создадим новый фрейм
    t = Table(root, list1)

