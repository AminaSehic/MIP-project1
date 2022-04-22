import sys
from os import listdir
from os.path import isfile, join, splitext
from tkinter import Tk, filedialog, Button, Canvas, Label, Entry

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pydicom as pydicom
from PIL import Image, ImageTk


def is_dcm_file(filepath):
    return splitext(join(filepath))[-1] == '.dcm'


def load_dcm_images_from_folder(folder):
    filepaths = [f"{folder}/{f}" for f in listdir(folder) if (isfile(join(folder, f)) and is_dcm_file(join(folder, f)))]
    return list(map(lambda filepath: pydicom.dcmread(filepath), filepaths))


def create_3d_array(originals):
    img_shape = list(originals[0].pixel_array.shape)
    img_shape.append(len(originals))
    img3d = np.zeros(img_shape)
    for i, s in enumerate(originals):
        img2d = s.pixel_array
        img3d[:, :, i] = img2d
    return img3d


class Gui:
    def __init__(self, master):
        self.master = master
        self.create_widgets()

    def create_widgets(self):
        self.select = Button(self.master, text="select an image", command=self.select_image)
        self.select.pack()

        self.subimage_label = Label(self.master, text="Enter coordinates to crop (left, top, right, bottom)")
        self.subimage_label.pack()

        self.subimage_coordinates = Entry(self.master)
        self.subimage_coordinates.pack()

        self.crop = Button(self.master, text="Crop", command=self.crop_image)
        self.crop.pack()

        self.watershed_button = Button(self.master, text="segmentation of cropped image", command=self.segmentation)
        self.watershed_button.pack()

        self.canvas = Canvas(self.master, width=532, height=532, bg="grey")
        self.canvas.pack()

    def load_dcm(self, file_path):
        return pydicom.dcmread(file_path)

    def select_image(self):
        file_path = filedialog.askopenfilename()
        image = self.load_dcm(file_path)
        image_pixels = image.pixel_array
        # windowing image to 0-255 pixel value range
        windowed_image = (np.maximum(image_pixels, 0) / image_pixels.max()) * 255.0
        self.image = Image.fromarray(windowed_image).resize((512, 512))
        bg_image = ImageTk.PhotoImage(self.image)
        self.canvas.bg_image = bg_image
        self.canvas.create_image(10, 10, image=self.canvas.bg_image, anchor='nw')

    def crop_image(self):
        left, top, right, bottom = map(lambda x: int(x), self.subimage_coordinates.get().split(','))
        try:
            self.image = self.image.crop((left, top, right, bottom)).resize((512, 512))
            bg_image = ImageTk.PhotoImage(self.image)
            self.canvas.bg_image = bg_image
            self.canvas.create_image(10, 10, image=self.canvas.bg_image, anchor='nw')
        except Exception as e:
            print(e)

    def segmentation(self):
        image = np.asarray(self.image).astype('uint8')
        # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # noise removal
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

        # sure background area
        sure_bg = cv2.dilate(opening, kernel, iterations=3)

        # Finding sure foreground area
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

        # Finding unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)
        fig, axs = plt.subplots(2, 2)
        axs[0, 0].imshow(thresh)
        axs[0, 0].set_title("threshold")
        axs[0, 1].imshow(sure_bg)
        axs[0, 1].set_title("sure background")
        axs[1, 0].imshow(sure_fg)
        axs[1, 0].set_title("sure foreground")
        axs[1, 1].imshow(unknown)
        axs[1, 1].set_title('Unknown')
        plt.show()

        ret, markers = cv2.connectedComponents(sure_fg)

        # Add one to all labels so that sure background is not 0, but 1
        markers = markers + 1

        # Now, mark the region of unknown with zero
        markers[unknown == 255] = 0
        color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        markers = cv2.watershed(color_image, markers)
        color_image[markers == -1] = [255, 0, 0]

        plt.imshow(color_image)
        plt.title("Image after watershed")
        plt.show()


def main(argv):
    # draw the gui with simple file/folder picker
    root = Tk()
    Gui(root)
    root.mainloop()


if __name__ == '__main__':
    main(sys.argv)
