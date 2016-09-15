#!/usr/bin/env python

import Tkinter
import sys

def button_on_clicked():
    print("Pushed button!")

def label_on_pressed(event):
    print("See you!")
    sys.exit()

window = Tkinter.Tk()

label = Tkinter.Label(window, text = "Label Sample")
label.pack(side = Tkinter.BOTTOM)
label.bind("<Button-1>", label_on_pressed)

button = Tkinter.Button(window, text = "Button Sample", command = button_on_clicked)
button.pack()

check_button = Tkinter.Checkbutton(window, text = "CheckButton Sample")
check_button.pack()

entry = Tkinter.Entry(window)
entry.insert(Tkinter.END, "Entry Sample")
entry.pack()

frame = Tkinter.LabelFrame(window, text= " LabelFrame Sample")
frame.pack()
Tkinter.Label(frame, text = "Label in Frame Sample").pack()

listbox = Tkinter.Listbox(window, height = 3)
listbox.insert(Tkinter.END, "Listbox Sample")
listbox.insert(Tkinter.END, "hoge")
listbox.pack()

scale = Tkinter.Scale(window, orient = Tkinter.HORIZONTAL)
scale.pack()

spinbox = Tkinter.Spinbox(window)
spinbox.pack()

window.mainloop()
