

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