from __future__ import division, absolute_import, print_function
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import *
import numpy as np
import matplotlib.pyplot as plt
from pyeit.mesh import create
import pyeit.eit.protocol as protocol
import pyeit.eit.greit as greit

"""1)Необходимо дописать функцию, которая будет брать измерительные кадры после усреднения - почти сделал
    2)Написать функцию, которая будет поворачивать изображение легких - сделано
    3)Написать функцию, которая будет отображать несколько графиков в одном - сделано"""

def get_breath_list(file_path: str) -> list[list[float]]:
    try:
        breath_file = open(file_path)
        breath_list = [line.split("	") for line in breath_file]
    except Exception:
        print('Error in get_breath_list function')  # Вывод информации об ошибке
    finally:
        breath_file.close()  # закрытие файла
    return breath_list

def get_rotated_list(breath_list: list[list[float]], rotate_num: int) -> list[list[int]]:
    """Функция сдвигает список с измерениями для того, чтобы крутить изображение легких"""
    try:
        result = []
        rotate_num = rotate_num % len(breath_list[0])
        for frame in breath_list:
            # frame = frame[rotate_num:] + frame[:rotate_num]
            result.append(frame[rotate_num:] + frame[:rotate_num])
    except Exception:
        print("Error in get_rotated_list function")
    finally:
        return result

def breath(file_path: str, num_of_frames: int):
    try:
        breath_matrix = get_breath_list(file_path)
        breath_points = []
        x = [i for i in range(0, 450, num_of_frames)]
        min_value_index = 0
        max_value_index = 0
        min_value = 1
        max_value = 0
        for i in range(0, len(breath_matrix), num_of_frames):
            summa = 0
            for k in range(num_of_frames):
                for j in breath_matrix[i + k]:
                    summa += float(j)
            average = summa / 208 / num_of_frames
            if average < min_value:
                min_value = average
                min_value_index = i
            if average > max_value:
                max_value = average
                max_value_index = i
            breath_points.append(average)
    except Exception:
        print('Error in breath function') # Вывод информации об ошибке
    # finally:
        # breath_file.close() # закрытие файла
    return [x, breath_points, min_value_index, max_value_index]

def get_framed_list(file_path: str, num_of_frames: int) -> [[]]:
    """Функция вернет раскадровку списка с измерениями"""
    breath_matrix = get_breath_list(file_path) # получаем матрицу дыхания
    result = []
    for i in range(0, len(breath_matrix), num_of_frames):
        breath_points = []  # переменная для хранения склеенных кадров
        for j in range(len(breath_matrix[0])):
            sum = 0 # переменная для хранения склеиваемых значений напряжения
            for k in range(num_of_frames):
                sum = sum + breath_matrix[i+k][j]
            breath_points.append(sum/num_of_frames)
        result.append(breath_points)
    return result

def get_framed_breath(file_path, num_of_frames):
    try:
        breath_matrix = get_framed_list(file_path, num_of_frames)
        breath_points = []
        x = [i for i in range(0, len(breath_matrix)*num_of_frames, num_of_frames)]
        min_value_index = 0
        max_value_index = 0
    except Exception:
        print('Error in breath function') # Вывод информации об ошибке
    finally:
        return [x, breath_points, min_value_index, max_value_index]


def build_all_graphs(root: tk.Tk, path, number_of_frames, coord):
    """Необходимо написать функцию так, чтобы каждый раз добавлялся новый массив координат
    и рисовался очередной график"""
    x, breath_points = get_coord_list(number_of_frames)
    figure3 = plt.Figure(figsize=(5, 4), dpi=100)
    ax3 = figure3.add_subplot(111)
    coord = coord + [x, breath_points, 'b']
    ax3.plot(*coord)
    scatter3 = FigureCanvasTkAgg(figure3, root)
    scatter3.get_tk_widget().pack(fill=BOTH)
    ax3.legend(['средняя разница напряжений'])
    ax3.set_xlabel('измерительные точки')
    ax3.set_ylabel('напряжение(В)')
    ax3.set_title('График дыхания')
    return coord

def get_coord_list(number_of_frames):
    """Эта функция нужна для того, чтобы вернуть координаты нового графика
    для общего графика"""
    if isinstance(number_of_frames, str):
        try:
            number_of_frames = int(number_of_frames)
        except Exception:
            print('Введите числовое значение', number_of_frames)
            number_of_frames = 1  # дефолтное значение
        finally:
            x, breath_points, ind_min, ind_max = breath('experimental.txt', number_of_frames)
            print("максимальный вдох имеет индекс", ind_max)
            print("минимальный выдох имеет индекс", ind_min)
            return [x, breath_points]

def build_graph(root: tk.Tk, path, number_of_frames):
    if isinstance(number_of_frames, str):
        try:
            number_of_frames = int(number_of_frames)
        except Exception:
            print('Введите числовое значение', number_of_frames)
            number_of_frames = 1  # дефолтное значение
        finally:
            x, breath_points, ind_max, ind_min = breath('experimental.txt', number_of_frames)
            figure3 = plt.Figure(figsize=(5, 4), dpi=100)
            ax3 = figure3.add_subplot(111)
            ax3.plot(x, breath_points, color='b')
            scatter3 = FigureCanvasTkAgg(figure3, root)
            scatter3.get_tk_widget().pack(fill=BOTH)
            ax3.legend(['средняя разница напряжений'])
            ax3.set_xlabel('измерительные точки')
            ax3.set_ylabel('напряжение(В)')
            ax3.set_title('График дыхания')

class Table:
    def __init__(self, root: tk.Frame, lst: list[set]):
        # code for creating table
        total_rows = len(lst)
        total_columns = len(lst[0])
        for i in range(total_rows):
            for j in range(total_columns):
                self.e = tk.Entry(root, width=20, fg='blue',
                                  font=('Arial', 16, 'bold'))
                self.e.grid(row=i, column=j)
                self.e.insert(tk.END, lst[i][j])

class AllMath():  # написал подсчет дисперсии относительно вообще всей информации в таблице
    @staticmethod
    def expected_value(path):  # расчет матожидания (M)
        """Расчитывает матожидание, возвращает список, первый элемент - матожидание,
         второй элемент - список с параметрами дыхания"""
        try:
            breath_file = open(path)
            breath_matrix = [line.split("	") for line in breath_file]
            summa = 0
            for line in breath_matrix:
                for value in line:
                    summa += float(value)
            result = summa / (len(breath_matrix) * len(breath_matrix[0]))
        except Exception:
            print('Ошибка в вычислении математического ожидания')
        finally:
            breath_file.close()
        return [result, breath_matrix]

    @staticmethod
    def dispersion(path):  # расчет дисперсии (D)
        try:
            average, breath_matrix = AllMath.expected_value(path)
            summa = 0
            for line in breath_matrix:
                for value in line:
                    summa += ((float(value) - average) ** 2)
            result = summa / (len(breath_matrix) * len(breath_matrix[0]))
        except Exception:
            print('Ошибка в вычислении дисперсии')
        return result
    @staticmethod
    def standard_deviation(path):  # расчет среднеквадратического отклонения(s0)
        return AllMath.dispersion(path) ** (1 / 2)

def make_table(root: tk.Tk, num_of_frames):
    if num_of_frames == '':
        num_of_frames = 1
    list1 = [("Количество кадров n", "max", "min", "матожидание M(x)", "дисперсия D(x)", "СКО s0"),
             (num_of_frames, "max", "min", AllMath.expected_value('experimental.txt')[0],
              AllMath.dispersion('experimental.txt'), AllMath.standard_deviation('experimental.txt'))]
    # создадим новый фрейм
    t = Table(root, list1)

def make_reconstruction(root: tk.Tk, number_of_frames):
    if number_of_frames == '':
        number_of_frames = 1
    x, breath_points, ind_min, ind_max = breath('experimental.txt', number_of_frames)
    # print("Вывожу breath_points")
    # [print() for i in range(10)]
    # print(breath_points)
    min_index = 11 # вот этот индекс нужно высчитывать и передавать в функцию реконструкции изображения
    max_index = 339 # этот параметр будет передан
    # необходимо пересчитывать максимальный и минимальный индекс

    n_el = 16
    mesh_obj = create(n_el, h0=0.05)

    file_path = 'experimental.txt'

    # breath_file = open('experimental.txt') # нужно не файл открывать, а передавать список как аргумент функции
    # breath_matrix = [line.split("	") for line in breath_file] # вместо вот этого
    breath_matrix = breath(file_path)
    # print("Вывожу breath_matrix")
    # [print() for i in range(10)]
    # print(breath_matrix)
    # print(breath_points==breath_matrix)
    # print("Тип breath_matrix",type(breath_matrix))
    # print("Тип breath_points",type(breath_points))
    v1 = np.array([float(i) for i in breath_matrix[max_index]])  # жестко заданы максимальный(вдох)
    v0 = np.array([float(i) for i in breath_matrix[min_index]])  # и минимальный(выдох) кадры дыхания

    """ 2. FEM forward simulations """
    # setup EIT scan conditions
    protocol_obj = protocol.create(n_el, dist_exc=1, step_meas=1,
                                   parser_meas="std")  # что за три параметра после кол-ва электродов?

    """ 3. Construct using GREIT """
    eit = greit.GREIT(mesh_obj, protocol_obj)  # создается обьект eit класса greit c помощью метода GREIT
    eit.setup(p=0.50, lamb=0.01, perm=1, jac_normalized=True)
    ds = eit.solve(v1, v0, normalize=True)  # нужно присмотреться к методу solve, он решит мои проблемы
    x, y, ds = eit.mask_value(ds, mask_value=np.NAN)  # что такое Any и NAN?

    # show alpha
    fig, axes = plt.subplots(constrained_layout=True, figsize=(6, 9))  # что делает метод subplots?

    # plot
    scatter3 = FigureCanvasTkAgg(fig, root)
    scatter3.get_tk_widget().pack(fill=BOTH)  # как приладить картинку сбоку?

    im = axes.imshow(np.real(ds), interpolation="none", cmap=plt.cm.viridis)

    # fig.colorbar(im, ax=axes.ravel().tolist())

    # нужно добавить функцию для отображения нескольких графиков на одном для сравнения
