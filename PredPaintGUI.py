from keras.models import load_model
from editor import process_image, find_magic_points
import numpy as np
import tkinter as tk
from parameters import Parameters, State


class PredPaintGUI:

    def __init__(self):

        # root parts:
        self.change_model_button = None
        self.drop_down = None
        self.chosen_model = None
        self.models = ['ADAM_model.h5', 'RMSprop_model.h5', 'ADAM_BIG_model.h5', 'RMS_LOW_DROPOUT_model.h5']
        self.model = load_model('models/ADAM_model.h5')
        self.header = None
        self.entry = None
        self.header_label = None
        self.footer = None
        self.footer_label = None
        self.footer_info = None
        self.footer_arrow = None

        # main parts:
        self.root = tk.Tk()
        self.set_header()

        self.cnv = tk.Canvas(self.root, width=Parameters.WIDTH, height=Parameters.HEIGHT, bg="white")
        self.set_canvas()

        self.set_footer()

        # image info
        self.index = 0
        self.info_about_photo = []  # [(x, y, index) ]
        self.image_size = {"top": Parameters.HEIGHT, "bottom": -1, "left": Parameters.WIDTH, "right": -1}

        # other auxiliary parts:
        self.state = State.PAINING
        self.previous_event = None

    def set_canvas(self):

        self.cnv.pack()
        self.cnv.bind("<B1-Motion>", self.paint_circle)
        self.cnv.bind("<ButtonRelease-1>", self.add_img)

    def set_header(self):

        self.root.title("Symbol Recognition")
        self.root.geometry(str(Parameters.WIDTH + 10) + "x" + str(Parameters.HEIGHT + 130))

        self.header = tk.Frame(self.root)
        self.header.columnconfigure(0, weight=1)

        self.header_label = tk.Label(self.header, text="Choose model: ", font=("Arial", 18))
        self.header_label.configure(padx=10, pady=10)
        self.header_label.grid(row=0, column=0)

        # Make a drop-down menu
        self.chosen_model = tk.StringVar(self.header)
        self.chosen_model.set(self.models[0])  # default value
        self.drop_down = tk.OptionMenu(self.header, self.chosen_model, *self.models)
        self.drop_down.grid(row=0, column=1)

        # Make a button to change the model and pass the current model to the function
        self.change_model_button = tk.Button(self.header, text="Change model", command=lambda: self.change_model(self.chosen_model.get()))
        self.change_model_button.grid(row=0, column=2)
        self.header.pack()

    def set_footer(self):

        self.footer = tk.Frame(self.root)
        self.footer.configure(pady=10, padx=5)
        self.footer.columnconfigure(0, weight=1)
        self.footer.columnconfigure(1, weight=1)
        self.footer.columnconfigure(2, weight=1)

        self.footer_info = tk.Label(self.footer, text="Predicted class: ", font=("Arial", 18), anchor="w")
        self.footer_info.grid(row=0, column=0, sticky=tk.W + tk.E)

        self.footer.pack(fill="x")

    def paint_circle(self, event):

        if self.state == State.PAINING:

            r = Parameters.RADIUS
            x, y = event.x, event.y

            if not self.if_brush_on_canvas(x, y, r):
                return

            self.change_image_size(x, y)

            if self.previous_event is not None:
                self.interpolation(x, y, r)

            self.cnv.create_oval(x - r, y - r, x + r, y + r, fill='black')

            self.info_about_photo.append((x, y, self.index))

            self.index += 1
            self.previous_event = event

    def interpolation(self, x, y, r):

        x_prev = self.previous_event.x
        y_prev = self.previous_event.y

        points = find_magic_points(x, y, x_prev, y_prev, r)

        if points is not None:
            self.cnv.create_polygon(points)

    def change_image_size(self, x, y):

        if y < self.image_size["top"]:
            self.image_size["top"] = y

        if y > self.image_size["bottom"]:
            self.image_size["bottom"] = y

        if x < self.image_size["left"]:
            self.image_size["left"] = x

        if x > self.image_size["right"]:
            self.image_size["right"] = x

    def restart_image_size(self):

        self.image_size["top"] = Parameters.HEIGHT
        self.image_size["bottom"] = 0
        self.image_size["left"] = Parameters.WIDTH
        self.image_size["right"] = 0

    def add_img(self, event):

        if self.state == State.PAINING:

            if not self.image_exits():
                print("You create blank image. It will not be added")
                return

            np_img = process_image(self.info_about_photo, self.image_size, Parameters.RADIUS)
            np_img = np.expand_dims(np_img, axis=0)
            index_to_label = {0: "circle", 1: "triangle", 2: "lightning", 3: "reversedS", 4: "S", 5: "W"}
            # predict the symbol, the model accepts 32x32 images which the image already is
            prediction = self.model.predict(np_img)

            self.cnv.delete("all")
            self.index = 0
            self.info_about_photo = []
            self.previous_event = None
            self.restart_image_size()
            self.footer_info.configure(text="Predicted class: " + index_to_label[np.argmax(prediction)]
                                            + " with probability: " + str(np.max(prediction)))

            print("image has been added")

        else:
            print("Set The Symbol CLass!")

    def image_exits(self):

        if len(self.info_about_photo) > 0:
            return True

        return False

    @staticmethod
    def if_brush_on_canvas(x, y, r):

        if 0 <= x - r and x + r <= Parameters.WIDTH - 1 and 0 <= y - r and y + r < Parameters.HEIGHT - 1:
            return True

        return False

    def start(self):
        self.root.mainloop()

    def change_model(self, new_model):
        models_path = "models/"
        self.model = load_model(models_path + new_model)
        print("Model changed to " + new_model)
