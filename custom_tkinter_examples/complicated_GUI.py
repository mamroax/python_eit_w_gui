import tkinter
import tkinter.messagebox
import customtkinter
import functions
from tkinter import *

customtkinter.set_appearance_mode("Dark")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("dark-blue")  # Themes: "blue" (standard), "green", "dark-blue"

class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        # configure window
        self.title("Графический интерфейс для Электоимпедансной томографии (ЭИТ)")
        self.geometry(f"{1100}x{580}")

        # configure grid layout (4x4)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure((2, 3), weight=0)
        self.grid_rowconfigure((0, 1, 2), weight=1)

        # create sidebar frame with widgets
        self.sidebar_frame = customtkinter.CTkFrame(self, width=140, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(4, weight=1)
        self.logo_label = customtkinter.CTkLabel(self.sidebar_frame, text="Настройки", font=customtkinter.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))
        self.sidebar_button_1 = customtkinter.CTkButton(self.sidebar_frame, command=self.sidebar_button_event)
        self.sidebar_button_1.grid(row=1, column=0, padx=20, pady=10)
        self.sidebar_button_2 = customtkinter.CTkButton(self.sidebar_frame, command=self.sidebar_button_event)
        self.sidebar_button_2.grid(row=2, column=0, padx=20, pady=10)
        self.sidebar_button_3 = customtkinter.CTkButton(self.sidebar_frame, command=self.sidebar_button_event)
        self.sidebar_button_3.grid(row=3, column=0, padx=20, pady=10)
        self.sidebar_button_4 = customtkinter.CTkButton(self.sidebar_frame, command=self.sidebar_button_event)
        self.sidebar_button_4.grid(row=4, column=0, padx=20, pady=10)
        self.appearance_mode_label = customtkinter.CTkLabel(self.sidebar_frame, text="Нажмите для переключения режима", anchor="w")
        self.appearance_mode_label.grid(row=5, column=0, padx=20, pady=(10, 0))
        self.appearance_mode_optionemenu = customtkinter.CTkOptionMenu(self.sidebar_frame, values=["Light", "Dark"],
                                                                       command=self.change_appearance_mode_event)
        self.appearance_mode_optionemenu.grid(row=6, column=0, padx=20, pady=(10, 10))
        self.scaling_label = customtkinter.CTkLabel(self.sidebar_frame, text="Размер текста и виджетов:", anchor="w")
        self.scaling_label.grid(row=7, column=0, padx=20, pady=(10, 0))
        self.scaling_optionemenu = customtkinter.CTkOptionMenu(self.sidebar_frame, values=["50%", "60%", "70%", "80%",
                                                                                           "90%", "100%", "110%",
                                                                                           "120%"],
                                                               command=self.change_scaling_event)
        self.scaling_optionemenu.grid(row=8, column=0, padx=20, pady=(10, 20))

        # create main entry and button
        self.entry = customtkinter.CTkEntry(self, placeholder_text="Введите целое положительное число кадров для усреднения")
        self.entry.grid(row=3, column=1, columnspan=2, padx=(20, 0), pady=(20, 20), sticky="nsew")


        self.main_button_1 = customtkinter.CTkButton(master=self, command=self.main_button1_event,
                                                     text="Рассчитать",
                                                     fg_color="transparent", border_width=2,
                                                     text_color=("gray10", "#DCE4EE"))
        self.main_button_1.grid(row=3, column=3, padx=(20, 20), pady=(20, 20), sticky="nsew")

        # making a frame for adding charts
        # self.chart_frame = customtkinter.CTkFrame(self)
        # self.chart_frame.grid(row=0, column=1, padx=(10,10), pady=(10, 10), sticky="nsew")

        # # create scrollable frame
        # self.scrollable_frame = customtkinter.CTkScrollableFrame(self, label_text="Настройки для визуализации")
        # self.scrollable_frame.grid(row=1, column=2, padx=(20, 0), pady=(20, 0), sticky="nsew")
        # self.scrollable_frame.grid_columnconfigure(0, weight=1)
        # self.scrollable_frame_switches = []
        # for i in range(100):
        #     switch = customtkinter.CTkSwitch(master=self.scrollable_frame, text=f"Параметр {i}")
        #     switch.grid(row=i, column=0, padx=10, pady=(0, 20))
        #     self.scrollable_frame_switches.append(switch)

        self.frame = Frame(master=self)
        self.frame.grid(row=0, column=1, rowspan=1, columnspan=3, padx=(20, 20), pady = (20, 20), sticky = "nsew")

        self.scrollable_frame = customtkinter.CTkScrollableFrame(self, label_text="Визуализация")
        self.scrollable_frame.grid(row=1, column=1, rowspan=2, columnspan=3, padx=(20, 20), pady=(20, 20), sticky="nsew")
        self.scrollable_frame_switches = []

        # set default values
        self.sidebar_button_1.configure(state="enabled", text="Первое окно")
        self.sidebar_button_2.configure(state="enabled", text="Второе окно")
        self.sidebar_button_3.configure(state="enabled", text="Третье окно")
        self.sidebar_button_4.configure(state="disabled", text="Вывести все графики в одном окне")
        self.appearance_mode_optionemenu.set("Dark")
        self.scaling_optionemenu.set("100%")

    def open_input_dialog_event(self):
        dialog = customtkinter.CTkInputDialog(text="Type in a number:", title="CTkInputDialog")
        print("CTkInputDialog:", dialog.get_input())

    def change_appearance_mode_event(self, new_appearance_mode: str):
        customtkinter.set_appearance_mode(new_appearance_mode)

    def change_scaling_event(self, new_scaling: str):
        new_scaling_float = int(new_scaling.replace("%", "")) / 100
        customtkinter.set_widget_scaling(new_scaling_float)

    def sidebar_button_event(self):
        print("sidebar_button click")

    def delete_frame(self):
        print("удаляю фреймбокс")
        # print(self.scrollable_frame_switches)
        # print(self.scrollable_frame)
        # self.scrollable_frame_switches = []
        self.scrollable_frame.destroy()
        print(self.scrollable_frame.grid_info())
        self.scrollable_frame = customtkinter.CTkScrollableFrame(self, label_text="Визуализация")
        self.scrollable_frame.grid(row=0, column=1, rowspan=3, columnspan=3, padx=(20, 20), pady=(20, 20), sticky="nsew")

    def main_button1_event(self):
        """Создание обьекта, содержащего график и таблицу с математическими рассчетами"""
        # ДЛЯ НАЧАЛА ПРОСТО ВЫВЕДУ ГРАФИК
        # в скроллбар добавить получилось(наконец-то!)
        new_frame1 = Frame(master=self.scrollable_frame)
        new_frame1.pack(fill=X, side=TOP, padx=(20, 20), pady=(20, 20))
        new_frame2 = Frame(master=self.scrollable_frame)
        new_frame2.pack(fill=X,side=TOP, padx=(20, 20), pady=(20, 20))
        new_frame3 = Frame(master=self.scrollable_frame)
        new_frame3.pack(fill=X, side=TOP, padx=(20, 20), pady=(20, 20))

        self.scrollable_frame_switches.append(functions.build_graph(new_frame1, "какой-то путь", self.entry.get()))
        functions.build_all_graphs(self.frame, "какой-то путь", self.entry.get())

        # теперь нужно подумать как удалить выбранный график(сделано)
        # как приладить к графику справа таблицу?
        self.scrollable_frame_switches.append(
            functions.make_table(new_frame2, self.entry.get()))
        self.sidebar_button_1 = customtkinter.CTkButton(self.scrollable_frame, command=self.delete_frame, text="удалить")
        self.sidebar_button_1.pack(side=TOP, padx=20, pady=10)
        # добавить графики реконструированного дыхания с выводом названия алгоритма
        # вывод графиков дыхания раздельно для левого и правого легких
        # желательно сделать вывод всех графиков в одной строке последовательно с возможностью прокрутки
        # добавить функцию для вывода всех графиков в одном по нажатию кнопки
        # либо добавлять все графики сразу по умолчанию в одно окно
        self.scrollable_frame_switches.append(
            functions.make_reconstruction(new_frame3))

if __name__ == "__main__":
    app = App()
    app.mainloop()
