# importing tkinter gui
import tkinter as tk
from tkinter import *
import functions as f

# creating window
window = tk.Tk()

# getting screen width and height of display
width = window.winfo_screenwidth()
height = window.winfo_screenheight()

# setting tkinter window size
window.geometry("%dx%d" % (width, height))
window.title("Графическкий интерфейс для Электоимпедансной томографии (ЭИТ)")
label = tk.Label(window, text="Графики и расчетные значения")
label.pack()

# Create A Main frame
main_frame = Frame(window)
main_frame.pack(fill=BOTH,expand=1)

# Create Frame for X Scrollbar
sec = Frame(main_frame)
sec.pack(fill=X,side=TOP)

# создадим поле для ввода количества кадров усреднения
L1 = Label(sec, text="Введите количество кадров для усреднения") # Создаем подпись для пояснения (лейбл)
L1.pack(side = LEFT)
E1 = Entry(sec)
E1.pack(side = LEFT)

# # Create A Main frame
# main_frame = Frame(window)
# main_frame.pack(fill=BOTH,expand=1)
#
# # Create Frame for X Scrollbar
# sec = Frame(main_frame)
# sec.pack(fill=X,side=BOTTOM)
#
# # Create A Canvas
# my_canvas = Canvas(main_frame)
# my_canvas.pack(side=LEFT,fill=BOTH,expand=1)
#
# # Add A Scrollbars to Canvas
# x_scrollbar = Scrollbar(sec,orient=HORIZONTAL,command=my_canvas.xview)
# x_scrollbar.pack(side=BOTTOM,fill=X)
# y_scrollbar = Scrollbar(main_frame,orient=VERTICAL,command=my_canvas.yview)
# y_scrollbar.pack(side=RIGHT,fill=Y)
#
# # Configure the canvas
# my_canvas.configure(xscrollcommand=x_scrollbar.set)
# my_canvas.configure(yscrollcommand=y_scrollbar.set)
# my_canvas.bind("<Configure>",lambda e: my_canvas.config(scrollregion= my_canvas.bbox(ALL)))
#
# # Create Another Frame INSIDE the Canvas
# second_frame = Frame(my_canvas)
#
# # Add that New Frame a Window In The Canvas
# my_canvas.create_window((0,0),window= second_frame, anchor="nw")
#
# for thing in range(100):
#     Button(second_frame ,text=f"Button  {thing}").grid(row=5,column=thing,pady=10,padx=10)
#
# for thing in range(100):
#     Button(second_frame ,text=f"Button  {thing}").grid(row=thing,column=5,pady=10,padx=10)

# # идея состоит в том, чтобы запихивать каждый новый график в скролбар
# # создадим скролбар
# scrollbar = Scrollbar(window)
# scrollbar.pack( side = RIGHT, fill = Y )
# mylist = Listbox(window, yscrollcommand = scrollbar.set )
# mylist.pack( side = LEFT)

def kek():
    # mylist.insert(f.build_graph(window))
    f.build_graph(main_frame, 'путь', int(E1.get()))

main_button = Button(text="Изменить", width=15, height=3)
main_button.config(command=kek)
main_button.pack()

window.mainloop() # это конечная команда для отображения окна