import matplotlib.pyplot as plt

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
    plt.plot(x, breath_points)
    plt.scatter(x, breath_points)
    plt.show()
    print('Индекс измерения с самым глубоким выдохом(момент времени tref) ', min_value_index)
    print('Индекс измерения с самым глубоким вдохом ', max_value_index)
except Exception:
    print('Error')
finally:
    breath_file.close()
