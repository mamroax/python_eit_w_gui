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
        self.title("Graphical Interface for Electrical Impedance Tomography (EIT)")
        self.geometry(f"{1100}x{580}")

        # configure grid layout (4x4)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure((2, 3), weight=0)
        self.grid_rowconfigure((0, 1, 2), weight=1)

        # create sidebar frame with widgets
        self.sidebar_frame = customtkinter.CTkFrame(self, width=140, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(4, weight=1)
        self.logo_label = customtkinter.CTkLabel(self.sidebar_frame, text="Settings", font=customtkinter.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))
        self.sidebar_button_1 = customtkinter.CTkButton(self.sidebar_frame, command=self.sidebar_button_event)
        self.sidebar_button_1.grid(row=1, column=0, padx=20, pady=10)
        self.sidebar_button_2 = customtkinter.CTkButton(self.sidebar_frame, command=self.sidebar_button_event)
        self.sidebar_button_2.grid(row=2, column=0, padx=20, pady=10)
        self.algoritm_label = customtkinter.CTkLabel(self.sidebar_frame, text="Choose a reconstruction algorithm",
                                                 font=customtkinter.CTkFont(size=20, weight="bold"))
        self.algoritm_label.grid(row=3, column=0, padx=20, pady=(20, 10))
        # self.sidebar_button_4 = customtkinter.CTkButton(self.sidebar_frame, command=self.sidebar_button_event)
        # self.sidebar_button_4.grid(row=4, column=0, padx=20, pady=10)
        self.appearance_mode_label = customtkinter.CTkLabel(self.sidebar_frame, text="Click to switch mode", anchor="w")
        self.appearance_mode_label.grid(row=5, column=0, padx=20, pady=(10, 0))
        self.appearance_mode_optionemenu = customtkinter.CTkOptionMenu(self.sidebar_frame, values=["Light", "Dark"],
                                                                       command=self.change_appearance_mode_event)
        self.appearance_mode_optionemenu.grid(row=6, column=0, padx=20, pady=(10, 10))
        self.algoritm_choice = customtkinter.CTkOptionMenu(self.sidebar_frame, values=["GREIT", "JAC"])
        self.algoritm_choice.grid(row=4, column=0, padx=20, pady=10)
        self.scaling_label = customtkinter.CTkLabel(self.sidebar_frame, text="Text and Widget Size:", anchor="w")
        self.scaling_label.grid(row=7, column=0, padx=20, pady=(10, 0))
        self.scaling_optionemenu = customtkinter.CTkOptionMenu(self.sidebar_frame, values=["50%", "60%", "70%", "80%",
                                                                                           "90%", "100%", "110%",
                                                                                           "120%"],
                                                               command=self.change_scaling_event)
        self.scaling_optionemenu.grid(row=8, column=0, padx=20, pady=(10, 20))

        # create main entry and button
        self.entry = customtkinter.CTkEntry(self, placeholder_text="Enter a positive integer number of frames to average")
        self.entry.grid(row=3, column=1, columnspan=2, padx=(20, 0), pady=(20, 20), sticky="nsew")


        self.main_button_1 = customtkinter.CTkButton(master=self, command=self.main_button1_event,
                                                     text="Calculate",
                                                     fg_color="transparent", border_width=2,
                                                     text_color=("gray10", "#DCE4EE"))
        self.main_button_1.grid(row=3, column=3, padx=(20, 20), pady=(20, 20), sticky="nsew")


        self.coord = [] # list for coordinates

        self.scrollable_frame = customtkinter.CTkScrollableFrame(self, label_text="Visualization")
        self.scrollable_frame.grid(row=1, column=1, rowspan=2, columnspan=3, padx=(20, 20), pady=(20, 20), sticky="nsew")
        self.scrollable_frame_switches = []

        # set default values
        self.sidebar_button_1.configure(state="enabled", text="First window")
        self.sidebar_button_2.configure(state="enabled", text="Second window")
        self.algoritm_choice.configure(state="enabled")
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
        print("deleting framebox")
        # print(self.scrollable_frame_switches)
        # print(self.scrollable_frame)
        # self.scrollable_frame_switches = []
        print(self.scrollable_frame.children)
        self.scrollable_frame.destroy()
        self.coord = []
        print(self.scrollable_frame.grid_info())
        self.scrollable_frame = customtkinter.CTkScrollableFrame(self, label_text="Visualization")
        self.scrollable_frame.grid(row=1, column=1, rowspan=2, columnspan=3, padx=(20, 20), pady=(20, 20), sticky="nsew")

    def main_button1_event(self):
        """Creating an object containing a graph and a table with mathematical calculations"""
        new_frame1 = Frame(master=self.scrollable_frame)
        new_frame1.pack(fill=X, side=TOP, padx=(20, 20), pady=(20, 20))
        new_frame2 = Frame(master=self.scrollable_frame)
        new_frame2.pack(fill=X,side=TOP, padx=(20, 20), pady=(20, 20))
        new_frame3 = Frame(master=self.scrollable_frame)
        new_frame3.pack(fill=X, side=TOP, padx=(20, 20), pady=(20, 20))
        new_frame4 = Frame(master=self.scrollable_frame)
        new_frame4.pack(fill=X, side=TOP, padx=(20, 20), pady=(20, 20))
        new_frame5 = Frame(master=self.scrollable_frame)
        new_frame5.pack(fill=X, side=TOP, padx=(20, 20), pady=(20, 20))
        new_frame6 = Frame(master=self.scrollable_frame)
        new_frame6.pack(fill=X, side=TOP, padx=(20, 20), pady=(20, 20))

        self.frame = Frame(master=self)
        self.frame.grid(row=0, column=1, rowspan=1, columnspan=3, padx=(20, 20), pady=(20, 20), sticky="nsew")
        self.scrollable_frame_switches.append(functions.build_graph(new_frame1, "path", self.entry.get()))
        self.coord = functions.build_all_graphs(self.frame, "path", self.entry.get(), self.coord)

        # self.scrollable_frame_switches.append(
        #     functions.make_table(new_frame2, self.entry.get()))
        # self.sidebar_button_1 = customtkinter.CTkButton(self.scrollable_frame, command=self.delete_frame, text="delete")
        # self.sidebar_button_1.pack(side=TOP, padx=20, pady=10)
        # self.scrollable_frame_switches.append(
        #     functions.make_reconstruction(new_frame3, self.entry.get()))
        # self.scrollable_frame_switches.append(
        #     functions.left_lung(new_frame4, self.entry.get()))
        # self.scrollable_frame_switches.append(
        #     functions.right_lung(new_frame5, self.entry.get()))
        self.scrollable_frame_switches.append(
            functions.make_jac(new_frame6, self.entry.get()))

if __name__ == "__main__":
    app = App()
    app.mainloop()
