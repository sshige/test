#!/usr/bin/env python
# downloaded from http://www.nslabs.jp/monkey-python-02b.rhtml

import Tkinter
from PIL import Image
import sys

class Scribble:
    def on_pressed(self, event):
        self.sx = event.x
        self.sy = event.y
        self.canvas.create_oval(self.sx, self.sy, event.x, event.y,
                                outline = self.color.get(),
                                width = self.width.get())

    def on_dragged(self, event):
        self.canvas.create_line(self.sx, self.sy, event.x, event.y,
                                fill = self.color.get(),
                                width = self.width.get())
        self.sx = event.x
        self.sy = event.y

    def on_pushed_quit(self):
        self.canvas.postscript(file="/tmp/tmp.eps")
        print("saved")
        sys.exit(0)

    def create_window(self):
        window = Tkinter.Tk()
        self.canvas = Tkinter.Canvas(window, bg = "white",
                                     width = 280, height = 280)
        self.canvas.pack()
        quit_button = Tkinter.Button(window, text = "Quit",
                                     command = self.on_pushed_quit)
        quit_button.pack(side = Tkinter.RIGHT)

        self.canvas.bind("<ButtonPress-1>", self.on_pressed)
        self.canvas.bind("<B1-Motion>", self.on_dragged)

        COLORS = ["red", "green", "blue", "#FF00FF", "black"]
        self.color = Tkinter.StringVar()
        self.color.set(COLORS[1])
        b = Tkinter.OptionMenu(window, self.color, *COLORS)
        b.pack(side = Tkinter.LEFT)

        self.width = Tkinter.Scale(window, from_ = 1, to = 15,
                                   orient = Tkinter.HORIZONTAL)
        self.width.set(5)
        self.width.pack(side = Tkinter.LEFT)

        return window;

    def __init__(self):
        self.window = self.create_window();

    def run(self):
        self.window.mainloop()

if __name__ == '__main__':
    Scribble().run()
