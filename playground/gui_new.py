# importing tkinter gui
import tkinter as tk
from tkinter import *
import functions

# creating window
window = tk.Tk()

# getting screen width and height of display
width = window.winfo_screenwidth()
height = window.winfo_screenheight()
# print(f"ширина окна {width}, высота окна {height}")

# setting tkinter window size
window.geometry("%dx%d" % (width, height))
window.title("Графическкий интерфейс для Электоимпедансной томографии (ЭИТ)") # задаем название окна
label = tk.Label(window, bg='lightblue', text="Графики и расчетные значения") # вешаем лейбл для пояснения
label.pack() # показать лейбл на экране, он выведется по центру эрана вверху(по умолчанию)

# Create A Main frame
top_frame = Frame(master=window) # создаем пустой контейнер под настройки, кнопки и поля
top_frame.pack(fill=BOTH,expand=1, side=TOP) # что такое fill, expand, side?

# Create Frame for X Scrollbar
top_frame_first = LabelFrame(top_frame, text="Настройки")
top_frame_first.pack(fill=X,side=TOP)

# создадим поле для ввода количества кадров усреднения
frames_number_label = Label(top_frame_first, bg='lightblue', text="Введите количество кадров для усреднения")
frames_number_label.pack(side = LEFT)
frames_number_box = Entry(top_frame_first, width=5)
frames_number_box.insert(0, "1")
frames_number_box.pack(side = LEFT)

path_label = Label(top_frame_first, bg='lightblue', text="Введите путь к файлу")
path_label.pack(side=LEFT)
path_box = Entry(top_frame_first)
path_box.insert(0, 'experimental.txt')
path_box.pack(side=LEFT)

# Create A Main frame
main_frame = Frame(window)
main_frame.pack(fill=BOTH,expand=1)

# Create Frame for X Scrollbar
sec = Frame(main_frame)
sec.pack(fill=X,side=BOTTOM)

# Create A Canvas
my_canvas = Canvas(main_frame)
my_canvas.pack(side=LEFT,fill=BOTH,expand=1)

# # Add A Scrollbars to Canvas
# x_scrollbar = Scrollbar(sec,orient=HORIZONTAL,command=my_canvas.xview)
# x_scrollbar.pack(side=BOTTOM,fill=X)
y_scrollbar = Scrollbar(main_frame,orient=VERTICAL,command=my_canvas.yview)
y_scrollbar.pack(side=RIGHT,fill=Y)

# Configure the canvas
# my_canvas.configure(xscrollcommand=x_scrollbar.set)
my_canvas.configure(yscrollcommand=y_scrollbar.set)
my_canvas.bind("<Configure>",lambda e: my_canvas.config(scrollregion= my_canvas.bbox(ALL)))

# Create Another Frame INSIDE the Canvas
second_frame = Frame(my_canvas)
#
# # Add that New Frame a Window In The Canvas
my_canvas.create_window((0,0),window= second_frame, anchor="nw")

# for thing in range(100):
#     Button(second_frame ,text=f"Button  {thing}").grid(row=5,column=thing,pady=10,padx=10)
#
# for thing in range(100):
#     Button(second_frame ,text=f"Button  {thing}").grid(row=thing,column=5,pady=10,padx=10)
#
# # идея состоит в том, чтобы запихивать каждый новый график в скролбар
# # создадим скролбар
# scrollbar = Scrollbar(window)
# scrollbar.pack( side = RIGHT, fill = Y )
# mylist = Listbox(window, yscrollcommand = scrollbar.set )
# mylist.pack( side = LEFT)

listbox = Listbox()
# вертикальная прокрутка
scrollbar = Scrollbar(orient="vertical", command = listbox.yview)

def kek():
    # mylist.insert(f.build_graph(window))
    functions.build_graph(second_frame, path_box.get(), int(frames_number_box.get()))

main_button = Button(top_frame, text="Изменить", width=15, height=3)
main_button.config(command=kek)
main_button.pack()

window.mainloop() # отображение окна
