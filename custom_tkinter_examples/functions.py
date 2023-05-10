from __future__ import division, absolute_import, print_function
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import *
import numpy as np
import matplotlib.pyplot as plt
from pyeit.mesh import create
import pyeit.eit.protocol as protocol
import pyeit.eit.greit as greit


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
    if isinstance(number_of_frames, str):
        try:
            number_of_frames = int(number_of_frames)
        except Exception:
            print('Введите числовое значение', number_of_frames)
            number_of_frames = 1 # дефолтное значение
        finally:
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
    def __init__(self, root:tk.Frame, lst:list[set]):
        # code for creating table
        total_rows = len(lst)
        total_columns = len(lst[0])
        for i in range(total_rows):
            for j in range(total_columns):
                self.e = tk.Entry(root, width=20, fg='blue',
                               font=('Arial', 16, 'bold'))
                self.e.grid(row=i, column=j)
                self.e.insert(tk.END, lst[i][j])


class AllMath(): # написал подсчет дисперсии относительно вообще всей информации в таблице
    @staticmethod
    def expected_value(path): # расчет матожидания (M)
        """Расчитывает матожидание, возвращает список, первый элемент - матожидание,
         второй элемент - список с параметрами дыхания"""
        try:
            breath_file = open(path)
            breath_matrix = [line.split("	") for line in breath_file]
            summa = 0
            for line in breath_matrix:
                for value in line:
                    summa+=float(value)
            result = summa/(len(breath_matrix)*len(breath_matrix[0]))
        except Exception:
            print('Ошибка в вычислении математического ожидания')
        finally:
            breath_file.close()
        return [result, breath_matrix]

    @staticmethod
    def dispersion(path): # расчет дисперсии (D)
        try:
            average, breath_matrix = AllMath.expected_value(path)
            summa = 0
            for line in breath_matrix:
                for value in line:
                    summa += ((float(value) - average)**2)
            result = summa / (len(breath_matrix) * len(breath_matrix[0]))
        except Exception:
            print('Ошибка в вычислении дисперсии')
        return result

    @staticmethod
    def standard_deviation(path): # расчет среднеквадратического отклонения(s0)
        return AllMath.dispersion(path) ** (1/2)


def make_table(root: tk.Tk, num_of_frames):
    if num_of_frames == '':
        num_of_frames = 1
    list1 = [("Количество кадров n", "max", "min", "матожидание M(x)", "дисперсия D(x)", "СКО s0"),
             (num_of_frames, "max", "min", AllMath.expected_value('experimental.txt')[0],
              AllMath.dispersion('experimental.txt'), AllMath.standard_deviation('experimental.txt'))]
    # создадим новый фрейм
    new_frame = Frame(master=root)
    new_frame.pack(fill=BOTH, side=TOP, padx=(20, 20), pady=(20, 20))
    t = Table(new_frame, list1)

def make_reconstruction(root: tk.Tk):
    """ 0. construct mesh """
    n_el = 16
    mesh_obj = create(n_el, h0=0.05)

    breath_file = open('experimental.txt')
    breath_matrix = [line.split("	") for line in breath_file]

    v1 = np.array([float(i) for i in breath_matrix[339]]) # жестко заданы максимальный(вдох)
    v0 = np.array([float(i) for i in breath_matrix[11]]) # и минимальный(выдох) кадры дыхания

    """ 2. FEM forward simulations """
    # setup EIT scan conditions
    protocol_obj = protocol.create(n_el, dist_exc=1, step_meas=1, parser_meas="std") # что за три параметра после кол-ва электродов?

    """ 3. Construct using GREIT """
    eit = greit.GREIT(mesh_obj, protocol_obj) # создается обьект eit класса greit c помощью метода GREIT
    eit.setup(p=0.50, lamb=0.01, perm=1, jac_normalized=True)
    ds = eit.solve(v1, v0, normalize=True) # нужно присмотреться к методу solve, он решит мои проблемы
    x, y, ds = eit.mask_value(ds, mask_value=np.NAN) # что такое Any и NAN?

    # show alpha
    fig, axes = plt.subplots(constrained_layout=True, figsize=(6, 9)) # что делает метод subplots?

    # plot
    scatter3 = FigureCanvasTkAgg(fig, root)
    scatter3.get_tk_widget().pack(side=tk.TOP) # как приладить картинку сбоку?

    im = axes.imshow(np.real(ds), interpolation="none", cmap=plt.cm.viridis)

    fig.colorbar(im, ax=axes.ravel().tolist())
