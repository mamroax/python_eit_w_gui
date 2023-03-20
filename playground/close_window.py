# Import the library
from tkinter import *

# Create an instance of window
win = Tk()

# Set the geometry of the window
win.geometry("700x350")

# Title of the window
win.title("Click the Button to Close the Window")

# Define a function to close the window
def close():
   #win.destroy()
   win.quit()

# Create a Button to call close()
Button(win, text= "Close the Window", font=("Calibri",14,"bold"), command=close).pack(pady=20)

win.mainloop()