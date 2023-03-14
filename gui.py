from tkinter import *
from tkinter import filedialog


def add_message():
    filename = filedialog.askopenfilename(initialdir="/", title="Выберите файл", \
                                          filetypes=(("txt файлы", "*.txt"), ("all files", "*.*")))
    with open(filename, 'r') as file:
        text = file.readline()
    message_entry.delete(0, END)
    message_entry.insert(0, text)


def save_message():
    filename = filedialog.asksaveasfilename(initialdir="/", title="Select file", \
                                            filetypes=(("txt файлы", "*.txt"), ("all files", "*.*")))
    with open(filename + '.txt', 'w') as file:
        file.write(message_entry.get())


root = Tk()
root.title("GUI на Python")
root.geometry("300x250")

message = StringVar()

message_entry = Entry(textvariable=message)
message_entry.place(relx=.5, rely=.1, anchor="c")

message_button = Button(text="Загрузить", command=add_message)
message_button.place(relx=.5, rely=.5, anchor="c")

message_button = Button(text="Сохранить", command=save_message)
message_button.place(relx=.5, rely=.7, anchor="c")

root.mainloop()