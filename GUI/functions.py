from __future__ import division, absolute_import, print_function
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import *
import numpy as np
import matplotlib.pyplot as plt
from pyeit.mesh import create
import pyeit.eit.protocol as protocol
import pyeit.eit.greit as greit

def get_breath_list(file_path: str) -> list[list[float]]:
    try:
        breath_file = open(file_path)
        breath_list = [line.split("	") for line in breath_file]
    except Exception:
        print('Error in get_breath_list function')
    finally:
        breath_file.close()
    return breath_list

def get_rotated_list(breath_list: list[list[float]], rotate_num: int) -> list[list[int]]:
    """The function shifts the list with measurements in order to rotate the image of the lungs
    (yet not working as it was intended to)"""
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
        print('Error in breath function')
    return [x, breath_points, min_value_index, max_value_index]

def get_framed_list(file_path: str, num_of_frames: int) -> [[]]:
    """The function will return a list storyboard with dimensions"""
    breath_matrix = get_breath_list(file_path) # get the breath matrix
    result = []
    for i in range(0, len(breath_matrix), num_of_frames):
        breath_points = []  # variable for storing glued frames
        for j in range(len(breath_matrix[0])):
            sum = 0 # variable for storing glued voltage values
            for k in range(num_of_frames):
                sum = sum + float(breath_matrix[i+k][j])
            breath_points.append(sum/num_of_frames)
        result.append(breath_points)
    return result

def get_framed_breath(file_path, num_of_frames):
    breath_matrix = get_framed_list(file_path, num_of_frames)
    x = [i for i in range(0, len(breath_matrix) * num_of_frames, num_of_frames)]
    min_value_index = 0
    max_value_index = 0
    max_value = 0
    min_value = 1
    # first we calculate the average in the list, but we will not remember it
    for i in range(len(breath_matrix)): # i - number of frame
        sum = 0 # variable to determine the sum of the frame
        for j in range(len(breath_matrix[i])): # j - this is the number of the measured potential difference in the frame
            sum = sum + float(breath_matrix[i][j])
        avg = sum / len(breath_matrix[i])
        if avg < min_value:
            min_value = avg
            min_value_index = i
        if avg > max_value:
            max_value = avg
            max_value_index = i
    return [x, breath_matrix, min_value_index, max_value_index]



def build_all_graphs(root: tk.Tk, path, number_of_frames, coord):
    """It is necessary to write a function so that a new array of coordinates is added each time
    and draw another graph"""
    x, breath_points = get_coord_list(number_of_frames)
    figure3 = plt.Figure(figsize=(5, 4), dpi=100)
    ax3 = figure3.add_subplot(111)
    coord = coord + [x, breath_points, 'b']
    ax3.plot(*coord)
    scatter3 = FigureCanvasTkAgg(figure3, root)
    scatter3.get_tk_widget().pack(fill=BOTH)
    ax3.legend(['average voltage difference'])
    ax3.set_xlabel('measuring points')
    ax3.set_ylabel('voltage(V)')
    ax3.set_title('Breath chart')
    return coord

def get_coord_list(number_of_frames):
    """This function is needed in order to return the coordinates of the new chart
    for general graph"""
    if isinstance(number_of_frames, str):
        try:
            number_of_frames = int(number_of_frames)
        except Exception:
            print('Enter a numeric value', number_of_frames)
            number_of_frames = 1  # дефолтное значение
        finally:
            x, breath_points, ind_min, ind_max = breath('experimental.txt', number_of_frames)
            print("maximum breath has an index", ind_max)
            print("minimum breath has an index", ind_min)
            return [x, breath_points]

def build_graph(root: tk.Tk, path, number_of_frames):
    if isinstance(number_of_frames, str):
        try:
            number_of_frames = int(number_of_frames)
        except Exception:
            print('Enter a numeric value', number_of_frames)
            number_of_frames = 1  # default value
        finally:
            x, breath_points, ind_max, ind_min = breath('experimental.txt', number_of_frames)
            figure3 = plt.Figure(figsize=(5, 4), dpi=100)
            ax3 = figure3.add_subplot(111)
            ax3.plot(x, breath_points, color='b')
            scatter3 = FigureCanvasTkAgg(figure3, root)
            scatter3.get_tk_widget().pack(fill=BOTH)
            ax3.legend(['average voltage difference'])
            ax3.set_xlabel('measuring points')
            ax3.set_ylabel('voltage(V)')
            ax3.set_title('Breath chart')

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

class AllMath(): # wrote a variance calculation relative to all information in the table
    @staticmethod
    def expected_value(path):  # calculation of Expected value (M)
        """Calculates the expected value, returns a list, the first element is the expected value,
         the second element is a list with breathing parameters"""
        try:
            breath_file = open(path)
            breath_matrix = [line.split("	") for line in breath_file]
            summa = 0
            for line in breath_matrix:
                for value in line:
                    summa += float(value)
            result = summa / (len(breath_matrix) * len(breath_matrix[0]))
        except Exception:
            print('Error in calculating the Expected value')
        finally:
            breath_file.close()
        return [result, breath_matrix]

    @staticmethod
    def dispersion(path):  # variance calculation (D)
        try:
            average, breath_matrix = AllMath.expected_value(path)
            summa = 0
            for line in breath_matrix:
                for value in line:
                    summa += ((float(value) - average) ** 2)
            result = summa / (len(breath_matrix) * len(breath_matrix[0]))
        except Exception:
            print('Error in calculating the variance')
        return result
    @staticmethod
    def standard_deviation(path):  # calculation of standard_deviation(s0)
        return AllMath.dispersion(path) ** (1 / 2)

def make_table(root: tk.Tk, num_of_frames):
    if num_of_frames == '':
        num_of_frames = 1
    list1 = [("Number of frames n", "max", "min", "Expected value M(x)", "variance D(x)", "standard deviation s0"),
             (num_of_frames, "max", "min", AllMath.expected_value('experimental.txt')[0],
              AllMath.dispersion('experimental.txt'), AllMath.standard_deviation('experimental.txt'))]
    t = Table(root, list1)

def make_reconstruction(root: tk.Tk, number_of_frames):
    if number_of_frames == '':
        number_of_frames = 1
    try:
        num_of_frames = int(number_of_frames)
    except Exception:
        print('Error in make_reconstruction - not an int number')
    x, breath_matrix, ind_min, ind_max = get_framed_breath('experimental.txt', num_of_frames)

    min_index = ind_min#11 # this index needs to be calculated and passed to the image reconstruction function
    max_index = ind_max#339 # this parameter will be passed
    # it is necessary to recalculate the maximum and minimum index

    n_el = 16
    mesh_obj = create(n_el, h0=0.05)

    file_path = 'experimental.txt'

    print("max index ", max_index)
    print("min index ", min_index)
    v1 = np.array([float(i) for i in breath_matrix[max_index]])  # жестко заданы максимальный(вдох)
    v0 = np.array([float(i) for i in breath_matrix[min_index]])  # и минимальный(выдох) кадры дыхания

    """ 2. FEM forward simulations """
    # setup EIT scan conditions
    protocol_obj = protocol.create(n_el, dist_exc=1, step_meas=1,
                                   parser_meas="std")  # что за три параметра после кол-ва электродов?

    """ 3. Construct using GREIT """
    eit = greit.GREIT(mesh_obj, protocol_obj)
    eit.setup(p=0.50, lamb=0.01, perm=1, jac_normalized=True)
    ds = eit.solve(v1, v0, normalize=True)
    x, y, ds = eit.mask_value(ds, mask_value=np.NAN)

    # show alpha
    fig, axes = plt.subplots(constrained_layout=True, figsize=(6, 9))

    # plot
    scatter3 = FigureCanvasTkAgg(fig, root)
    scatter3.get_tk_widget().pack(fill=BOTH)

    im = axes.imshow(np.real(ds), interpolation="none", cmap=plt.cm.viridis)

def left_lung():
    """function will return the reconstruction matrix for the left lung"""
    return 0

def right_lung():
    """function will return the reconstruction matrix for the right lung"""
    return 0
