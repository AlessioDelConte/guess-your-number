import io
import tkinter as tk

from PIL import Image


def draw(event):
    x, y = event.x, event.y
    if canvas.old_coords:
        x1, y1 = canvas.old_coords
        canvas.create_line(x, y, x1, y1)
    canvas.old_coords = x, y


def draw_line(event):
    if str(event.type) == 'ButtonPress':
        canvas.old_coords = event.x, event.y

    elif str(event.type) == 'Motion':
        draw(event)


def save_as_png(event):
    ps = canvas.postscript(colormode="mono")
    img = Image.open(io.BytesIO(ps.encode('utf-8')))
    img.save('number.png', 'png')
    root.destroy()


def draw_image():
    global root
    root = tk.Tk()

    global canvas
    canvas = tk.Canvas(root, width=200, height=200)
    canvas.pack()
    canvas.old_coords = None

    root.bind('<ButtonPress-1>', draw_line)
    root.bind('<B1-Motion>', draw_line)
    root.bind('<ButtonRelease-1>', save_as_png)

    root.mainloop()

