# import the necessary packages
from tkinter import *
from PIL import Image
from PIL import ImageTk
from tkinter import filedialog as tkFileDialog
import numpy as np
import cv2
from tkinter import messagebox as ms
from math import sqrt, exp
import  matplotlib.pyplot as py
import math

image = None
image1 = None
edged = None
mas_k = None


# region Important Function
def add_padding(img, k, type: False):
    """
    Add padding to the image.\n
    Add new rows and columns to the image
    :param img: input image.
    :param k: kernel size
    :param type: type of padding add zeros (False) or replicating the first and last row & column (True).
    :return: padded image
    """

    h, w, chs = np.shape(img)
    k_ = (k // 2)
    new_img = np.zeros((h + (k_ * 2), w + (k_ * 2), chs), np.uint8)
    if type:
        # fill new rows
        for ch in range(chs):
            for r in range(k_):
                for c in range(w):
                    # top new rows
                    new_img[r, c + k_, ch] = img[0, c, ch]
                    # bottom new rows
                    new_img[h + r + k_, c + k_, ch] = img[-1, c, ch]
                    pass
                pass
            pass

        # fill new columns
        for ch in range(chs):
            for r in range(h):
                for c in range(k_):
                    # left new columns
                    new_img[r + k_, c, ch] = img[r, 0, ch]
                    # right new columns
                    new_img[r + k_, w + c + k_, ch] = img[r, -1, ch]
                    pass
                pass
            pass

        # fil corners
        for ch in range(chs):
            for r in range(k_):
                for c in range(k_):
                    new_img[r, c, ch] = img[0, 0, ch]  # top left
                    new_img[r, c + w + k_, ch] = img[0, -1, ch]  # top right
                    new_img[r + h + k_, c + w + k_, ch] = img[-1, -1, ch]  # bottom right
                    new_img[r + h + k_, c, ch] = img[-1, 0, ch]  # bottom left
                    pass
                pass
            pass

        pass

    # fill original pixels
    for ch in range(chs):
        for r in range(h):
            for c in range(w):
                new_img[r + k_, c + k_, ch] = img[r, c, ch]
                pass
            pass
        pass

    return new_img


def aplay_filter(img, filter: np.ndarray,  type: bool, type_filter = True):
    """
    Apply filter in image
    :param img: the input image, should (RGB | gray)
    :param filter: filter matrix
    :param type: type of padding add zeros (False) or replicating the first and last row & column (True).
    :param type_filter: Arithmetic filter (True) | Geometric filter (False)
    :return: new image
    """

    # region Check on the parameters
    shape = np.shape(img)
    shape_fil = np.shape(filter)

    if shape_fil[0] != shape_fil[1]: raise "Filter should have equal rows and columns"
    if shape_fil[0] % 2 != 1: raise "Kernel Size should be odd number"
    try:
        # check in the third-dimension isn't correct for images shape
        if shape[2] != 3 & shape[2] != 1:
            mess = """"
        Image must have 3 dimension (h, w, ch).
        ch must
              1 for gray-level image
              3 for RGB image
        """
            raise TypeError(mess)

    except IndexError:
        # in case 2d array
        shape = shape + (1, )
        img = np.reshape(img, shape)
        pass
    # endregion
    h, w, chs = shape; k = shape_fil[0]

    def calculate_value_arth(window):
        """
        Calculate value of new adjacent pixel
        :param window: the pixels to which the calculation is applied.
        :return: new pixel value
        """
        value = 0
        for i in range(k):
            for j in range(k):
                value += window[i, j] * filter[i, j]
            pass
        if value < 0:return 0
        if value > 255: return 255
        return np.uint8(value)

    def calculate_value_geom(window):
        """
        Calculate value of new adjacent pixel
        :param window: the pixels to which the calculation is applied.
        :return: new pixel value
        """
        value = 1
        for i in range(k):
            for j in range(k):
                value *= window[i, j] * filter[i, j]
            pass

        # value = np.power(value, (1/ k ** 2)) # instance of this make filter contains the sqr of correlation.
        if value < 0:return 0
        if value > 255: return 255
        return np.uint8(value)

    img = img.copy()
    new_img = add_padding(img, k, type)

    if type_filter:
        for ch in range(chs):
            for r in range(h):
                for c in range(w):
                    img[r, c, ch] = calculate_value_arth(new_img[r: r+k, c: c+k, ch])
                    pass
                pass
            pass
        pass
    else :
        for ch in range(chs):
            for r in range(h):
                for c in range(w):
                    img[r, c, ch] = calculate_value_geom(new_img[r: r+k, c: c+k, ch])
                    pass
                pass
            pass
        pass

    return img
# endregion


"""____________________________________________________________________________"""

filters_need_img_size_number = ["unsharpen masking", "butterworth high pass",
                                "butterworth low pass"]
"""____________________________________________________________________________"""


def choice_need_img_size_number():
        global mas_k
    # try:
        if (listbox_need_img_size_number.get(listbox_need_img_size_number.curselection()) == "unsharpen masking"):
            mas_k = 33;
        if (listbox_need_img_size_number.get(listbox_need_img_size_number.curselection()) == "butterworth high pass"):
            mas_k = 34;
        if (listbox_need_img_size_number.get(listbox_need_img_size_number.curselection()) == "butterworth low pass"):
            mas_k = 35;
        select_image()
    # except:
    #     ms.showerror('Error!', 'You must choose a filter')


"""____________________________________________________________________________"""

filters_two_img = ["histogram matching", "add two images", "subtract two images"]#,
 #                  "watermark"]

"""____________________________________________________________________________"""


def choice_two_img():
        global mas_k
    # try:
        if (listbox_two_img.get(listbox_two_img.curselection()) == "histogram matching"):
            mas_k = 7;
        elif (listbox_two_img.get(listbox_two_img.curselection()) == "add two images"):
            mas_k = 8;
        elif (listbox_two_img.get(listbox_two_img.curselection()) == "subtract two images"):
            mas_k = 9;
        # elif (listbox_two_img.get(listbox_two_img.curselection()) == "watermark"):
        #     mas_k = 36;
        selectimage1()
        select_image()
    # except:
    #     ms.showerror('Error!', 'You must choose a filter')


"""____________________________________________________________________________"""
# "minimize"
filters_need_img_size = ["one order maximize", "Brightness", "power law",
                         "maximum filter", "mean filter", "median filter", "minimum filter",
                         "quantization", "weight filter", "point edg detiction filters",
                         "sharpening filter", "ideal high pass filter", "gaussian high pass filter",
                         "ideal low pass filter", "gaussian low pass filter", "geometric mean Filter",
                         "Mid-point Filter"
                         ]

"""____________________________________________________________________________"""


def choice_need_img_size():
        global mas_k
    # try:
        if (listbox_need_img_size.get(listbox_need_img_size.curselection()) == "one order maximize"):
            mas_k = 0;
        elif (listbox_need_img_size.get(listbox_need_img_size.curselection()) == "minimize"):
            mas_k = 1;
        elif (listbox_need_img_size.get(listbox_need_img_size.curselection()) == "Brightness"):
            mas_k = 3;
        elif (listbox_need_img_size.get(listbox_need_img_size.curselection()) == "power law"):
            mas_k = 5;
        elif (listbox_need_img_size.get(listbox_need_img_size.curselection()) == "maximum filter"):
            mas_k = 11;
        elif (listbox_need_img_size.get(listbox_need_img_size.curselection()) == "mean filter"):
            mas_k = 12;
        elif (listbox_need_img_size.get(listbox_need_img_size.curselection()) == "median filter"):
            mas_k = 13;
        elif (listbox_need_img_size.get(listbox_need_img_size.curselection()) == "minimum filter"):
            mas_k = 14;
        elif (listbox_need_img_size.get(listbox_need_img_size.curselection()) == "quantization"):
            mas_k = 15;
        elif (listbox_need_img_size.get(listbox_need_img_size.curselection()) == "weight filter"):
            mas_k = 16;
        elif (listbox_need_img_size.get(listbox_need_img_size.curselection()) == "point edg detiction filters"):
            mas_k = 21;
        elif (listbox_need_img_size.get(listbox_need_img_size.curselection()) == "sharpening filter"):
            mas_k = 26;
        elif (listbox_need_img_size.get(listbox_need_img_size.curselection()) == "ideal high pass filter"):
            mas_k = 27;
        elif (listbox_need_img_size.get(listbox_need_img_size.curselection()) == "gaussian high pass filter"):
            mas_k = 28;
        elif (listbox_need_img_size.get(listbox_need_img_size.curselection()) == "ideal low pass filter"):
            mas_k = 29;
        elif (listbox_need_img_size.get(listbox_need_img_size.curselection()) == "gaussian low pass filter"):
            mas_k = 30;
        elif (listbox_need_img_size.get(listbox_need_img_size.curselection()) == "geometric mean Filter"):
            mas_k = 31;
        elif (listbox_need_img_size.get(listbox_need_img_size.curselection()) == "Mid-point Filter"):
            mas_k = 32;

        select_image()
    # except:
    #     ms.showerror('Error!', 'You must choose a filter')


"""____________________________________________________________________________"""

filters_need_img = ["convert to gray", "contrast", "histogram equalization",
                    "image negative", "edge horizontal filter", "edge vertical Filter",
                    "edge diagonal\_Filter", "edge diagonal/_Filter", "Sharpening horizontal filter",
                    "Sharpening vertical filter", "Sharpening diagonal_r_Filter",
                    "Sharpening diagonal_l_Filter"
                    ]

"""____________________________________________________________________________"""


def choice_need_img():
        global mas_k
    # try:
        if (listbox_need_img.get(listbox_need_img.curselection()) == "convert to gray"):
            mas_k = 2;
        elif (listbox_need_img.get(listbox_need_img.curselection()) == "contrast"):
            mas_k = 4;
        elif (listbox_need_img.get(listbox_need_img.curselection()) == "histogram equalization"):
            mas_k = 6;
        elif (listbox_need_img.get(listbox_need_img.curselection()) == "image negative"):
            mas_k = 10;
        elif (listbox_need_img.get(listbox_need_img.curselection()) == "edge horizontal filter"):
            mas_k = 17;
        elif (listbox_need_img.get(listbox_need_img.curselection()) == "edge vertical Filter"):
            mas_k = 18;
        elif (listbox_need_img.get(listbox_need_img.curselection()) == "edge diagonal\_Filter"):
            mas_k = 19;
        elif (listbox_need_img.get(listbox_need_img.curselection()) == "edge diagonal/_Filter"):
            mas_k = 20;
        elif (listbox_need_img.get(listbox_need_img.curselection()) == "Sharpening horizontal filter"):
            mas_k = 22;
        elif (listbox_need_img.get(listbox_need_img.curselection()) == "Sharpening vertical filter"):
            mas_k = 23;
        elif (listbox_need_img.get(listbox_need_img.curselection()) == "Sharpening diagonal_r_Filter"):
            mas_k = 24;
        elif (listbox_need_img.get(listbox_need_img.curselection()) == "Sharpening diagonal_l_Filter"):
            mas_k = 25;

        select_image()
    # except:
    #     ms.showerror('Error!', 'You must choose a filter')


"""____________________________________________________________________________"""


def mask(mas_k, image):
    if mas_k == 0:
        return order1_max(image, int(sp.get()))
    elif mas_k == 1:
        return minimize(image, int(sp.get()))
    elif mas_k == 2:
        return gray_avg(image)
    elif mas_k == 3:
        return brightness(image, int(sp.get()))
    elif mas_k == 4:
        return contrast(image, 0, 255)
    elif mas_k == 5:
        return power_law(image, float(sp.get()))
    elif mas_k == 6:
        return equalization(image)
    elif mas_k == 7:
        return matching(image1, image)
    elif mas_k == 8:
        return image_algebra(image, image1,'+')
    elif mas_k == 9:
        return image_algebra(image, image1, '-')
    elif mas_k == 10:
        return negative(image)
    elif mas_k == 11:
        return max_Filter(int(sp.get()), image)
    elif mas_k == 12:
        return mean_Filter(int(sp.get()), image)
    elif mas_k == 13:
        return median_Filter(int(sp.get()), image)
    elif mas_k == 14:
        return mini_Filter(int(sp.get()), image)
    elif mas_k == 15:
        return quantization(image, int(sp.get()))
    elif mas_k == 16:
        return weight_Filter(int(sp.get()), image)
    elif mas_k == 17:
        return edg_horizontal_Filter(image)
    elif mas_k == 18:
        return edg_vertical_Filter(image)
    elif mas_k == 19:
        return edg_diagonal_r_Filter(image)
    elif mas_k == 20:
        return edg_diagonal_l_Filter(image)
    elif mas_k == 21:
        return edg_point_Filter(int(sp.get()), image)
    elif mas_k == 22:
        return Sharpening_horizontal_Filter(image)
    elif mas_k == 23:
        return Sharpening_vertical_Filter(image)
    elif mas_k == 24:
        return Sharpening_diagonal_r_Filter(image)
    elif mas_k == 25:
        return Sharpening_diagonal_l_Filter(image)
    elif mas_k == 26:
        return sharpening_Filter(int(sp.get()), image)
    elif mas_k == 27:
        return ideal_hpf(int(sp.get()), image)
    elif mas_k == 28:
        return gaussian_hpf(int(sp.get()), image)
    elif mas_k == 29:
        return ideal_lpf(int(sp.get()), image)
    elif mas_k == 30:
        return gaussian_lpf(int(sp.get()), image)
    elif mas_k == 31:
        return geometric_mean_Filter(int(sp.get()), image)
    elif mas_k == 32:
        return mid_point_Filter(int(sp.get()), image)
    elif mas_k == 33:
        return unsharpen_masking(int(sp2.get()), image, int(sp3.get()))
    elif mas_k == 34:
        return butterworth_hpf(int(sp2.get()), int(sp3.get()), image)
    elif mas_k == 35:
        return butterworth_lpf(int(sp2.get()), int(sp3.get()), image)
    elif mas_k == 36:
        return #watermark(image1, image)


"""_______________________________Mid-point Filter_________________________"""

def mid_point_Filter(img, k:int, type_padding = False):
    """
    Mid-Point is Order Restoration Filter, that used to remove noise from image
    :param img: the input Image, should be (RGB | gray)
    :param k: kernel size
    :param type_padding: type of padding add zeros (False) or replicating the first and last row & column (True).
    :return: new image.
    """

    # region Check on the parameters
    shape = np.shape(img)
    if k % 2 != 1: raise "Kernel Size should be odd number"
    try:
        # check in the third-dimension isn't correct for images shape
        if shape[2] != 3 & shape[2] != 1:
            mess = """"
        Image must have 3 dimension (h, w, ch).
        ch must
              1 for gray-level image
              3 for RGB image
        """
            raise TypeError(mess)

    except IndexError:
        # in case 2d array
        shape = shape + (1, )
        img = np.reshape(img, shape)
        pass
    # endregion

    img = img.copy()
    h, w, chs = np.shape(img)
    new_img = add_padding(img, k, type_padding)
    for ch in range(chs):
        for r in range(h):
            for c in range(w):
                # calculate mid-point value
                img[r, c, ch] = (np.min(new_img[r: r+k, c: c+k, ch]) + np.min(new_img[r: r+k, c: c+k, ch]) ) // 2
                pass
            pass
        pass

    return new_img

"""_________________________geometric mean Filter_____________________________"""


def geometric_mean_Filter(img, k:int, type_padding = False):
    """
    Geometry Mean is Mean Restoration Filter, that used to remove noise from image
    :param img: the input Image, should be (RGB | gray)
    :param k: kernel size
    :param type_padding: type of padding add zeros (False) or replicating the first and last row & column (True).
    :return: new image.
    """

    # region Check on the parameters
    shape = np.shape(img)
    if k % 2 != 1: raise "Kernel Size should be odd number"
    try:
        # check in the third-dimension isn't correct for images shape
        if shape[2] != 3 & shape[2] != 1:
            mess = """"
        Image must have 3 dimension (h, w, ch).
        ch must
              1 for gray-level image
              3 for RGB image
        """
            raise TypeError(mess)

    except IndexError:
        # in case 2d array
        shape = shape + (1, )
        img = np.reshape(img, shape)
        pass
    # endregion

    # apply average
    filter = np.ones((k, k))
    filter[:, :] = np.power(1, 1/ (k**2))
    return aplay_filter(img, filter, type_padding)

"""________________________________weight Filter_______________________________"""


def weight_Filter(img, siqma, type_padding = False):
    """
    Smoothing is useful, in remove noise pixels in images & remove small details include bridge small gaps in lines and
    extract objects.\n
    Smoothing is Neighbor Operation that enhance image by change the pixel value based on its adjacent (neighbor) pixels.
    :param img: the input image, should (RGB | gray)
    :param siqma: the degree of blurring kernel size
    :param type_padding: type of padding add zeros (False) or replicating the first and last row & column (True).
    :return: new smoothed image
    """

    k = 2 * (int(3.7 * (siqma - 0.5))) + 1
    filter =  np.zeros((k, k))

    siqma = siqma**2
    for i in range(k):
        for j in range(k):
            # apply Gaussian formula
            filter[i, j] = (1 / (2 * 3.14 * siqma)) * (2.718 ** (-1 * ( (i**2 + j**2) / (2 * siqma))))
            pass
        pass
    return aplay_filter(img, filter, type_padding)

"""_____________________________unsharpen masking______________________________"""


def unsharpen_masking(sigma,img,  type_padding = False, k = 3):
    """
    Sharping is useful to focus on high details
    :param img: the input image, should (RGB | gray)
    :param siqma: the degree of blurring kernel size
    :param type_padding: type of padding add zeros (False) or replicating the first and last row & column (True).
    :param k: the kernel size
    :return: new sharped image
    """

    return image_algebra(
        # subtract original image from smoothed image
        image_algebra(img,
                      weight_Filter(img, sigma, type_padding=type_padding), '-') # convert to smoothed image
        , img, '+') # sum original image anf output image


"""________________________________quantization Filter_______________________________"""


def quantization(img, k: int):
    """
    Image quantization is used to reduce the storage required to store the imag.\n
    Reduce the number of bits that store a pixel.\n
    Image Quantization is a Pixel Operation that enhances the image by changing in the intensity(value) of a pixel.
    :param img: the input image, should (gray | RGB)
    :param k: number of bits to store a pixel, represent the number of gray-levels in image.
    :return: new enhanced image
    """

    # region Check on the parameters
    shape = np.shape(img)
    if k < 1 | k > 8: raise "number of bits must between 1 and 8.\n available gray-levels"
    try:
        # check in the third-dimension isn't correct for images shape
        if shape[2] != 3 & shape[2] != 1:
            mess = """"
        Image must have 3 dimension (h, w, ch).
        ch must
              1 for gray-level image
              3 for RGB image
        """
            raise TypeError(mess)
    except IndexError:
        # in case 2d array
        shape = shape + (1, )
        img = np.reshape(img, shape)
        pass
    # endregion

    # region Code
    h, w, chs = shape
    # to write in image
    img = img.copy()

    # number of pixels in each group, number of gray-levels is 2 ** k.
    gap = 256 / (2 ** k)

    for ch in range(chs):
        for r in range(h):
            for c in range(w):
                # index of color in new space (gray-levels).
                i = int(img[r, c, ch] / gap)
                # replace pixel-color with corresponded color in new space,
                # that represent of a group of pixels.
                # the first pixel in group is represented of this group.
                img[r, c, ch] = i * gap
                pass
            pass
        pass
     # endregion

    return img

""""________________________________non linear smoothing ______________________"""


def smoothing_nonlinear(img, type_smoothing: int,  k = 3, type_padding = False):
    """
    Smoothing is useful, in remove noise pixels in images & remove small details include bridge small gaps in lines and
    extract objects.\n
    Smoothing is Neighbor Operation that enhance image by change the pixel value based on its adjacent (neighbor) pixels.
    :param img: the input image, should (RGB | gray)
    :param type_smoothing: type of non-linear filter, 0 (median) | 1 (min) | 2 (max)
    :param k: kernel size
    :param type_padding: type of padding add zeros (False) or replicating the first and last row & column (True).
    :return: new smoothed image
    """

    # region Check on the parameters
    shape = np.shape(img)
    if k % 2 != 1: raise "Kernel Size should be odd number"
    try:
        # check in the third-dimension isn't correct for images shape
        if shape[2] != 3 & shape[2] != 1:
            mess = """"
        Image must have 3 dimension (h, w, ch).
        ch must
              1 for gray-level image
              3 for RGB image
        """
            raise TypeError(mess)

    except IndexError:
        # in case 2d array
        shape = shape + (1, )
        img = np.reshape(img, shape)
        pass
    # endregion

    h, w, chs = shape

    img = img.copy()
    new_img = add_padding(img, k, type_padding)

    if type_smoothing == 0:
        for ch in range(chs):
            for r in range(h):
                for c in range(w):
                    img[r, c, ch] = np.median(new_img[r: r+k, c: c+k, ch])
                pass
            pass
        pass
    elif type_smoothing == 1:
        for ch in range(chs):
            for r in range(h):
                for c in range(w):
                    img[r, c, ch] = np.min(new_img[r: r+k, c: c+k, ch])
                pass
            pass
        pass
    elif type_smoothing == 2:
        for ch in range(chs):
            for r in range(h):
                for c in range(w):
                    img[r, c, ch] = np.max(new_img[r: r+k, c: c+k, ch])
                pass
            pass
        pass
    else: raise "type_smoothing Must 0 | 1 | 2"

    return img

"""________________________________minimum Filter_______________________________"""


def mini_Filter(mask_size, ori_img):
    return smoothing_nonlinear(ori_img, type_smoothing=1,  k = mask_size, type_padding = False)


"""_______________________mean Filter____________________________________"""


def mean_Filter(img, k = 3, type_padding = False):
    """
    Smoothing is useful, in remove noise pixels in images & remove small details include bridge small gaps in lines and
    extract objects.\n
    Smoothing is Neighbor Operation that enhance image by change the pixel value based on its adjacent (neighbor) pixels.
    :param img: the input image, should (RGB | gray)
    :param k: kernel size
    :param type_padding: type of padding add zeros (False) or replicating the first and last row & column (True).
    :return: new smoothed image
    """

    # apply average
    filter =  np.zeros((k, k))
    filter[:, :] = 1 / (k*k)
    return aplay_filter(img, filter, type_padding)


"""_______________________maximum Filter____________________________________"""


def max_Filter(mask_size, ori_img):
    return smoothing_nonlinear(ori_img, type_smoothing=2,  k = mask_size, type_padding = False)


"""___________________________ add two images _______________________________________"""


def image_algebra(first_img, second_image, type: str):
    """
    Apply algebra on 2-images
    :param first_img: the first image, should (gray | RGB)
    :param second_image: the second image, should (gray | RGB)
    :param type: + | -
    :return: new enhanced image
    """

    # region Check on the parameters
    shape_src = np.shape(first_img)
    shape_targ = np.shape(second_image)
    try:
        if (shape_src[0] != shape_targ[0]) | (shape_src[1] != shape_targ[1]):raise "Must have the same shape"
        # check in the third-dimension isn't correct for images shape
        if (shape_src[2] != 3 & shape_src[2] != 1) | (shape_targ[2] != 3 & shape_targ[2] != 1):
            mess = """"
        Image must have 3 dimension (h, w, ch).
        ch must
              1 for gray-level image
              3 for RGB image
        """
            raise TypeError(mess)
        if shape_src[2] != shape_targ[2] : raise "Must have the same shape"
    except IndexError:
        # in case 2d array
        shape_src = shape_src + (1, )
        first_img = np.reshape(first_img, shape_src)

        shape_targ = shape_targ + (1, )
        second_image = np.reshape(second_image, shape_targ)
        pass
    # endregion
    new_image = np.zeros(shape_src, np.uint8)
    h, w, chs = shape_src

    if type == '+':
         for ch in range(chs):
             for r in range(h):
                 for c in range(w):
                     pixel = int(first_img[r, c, ch]) + int(second_image[r, c, ch])

                     if pixel > 255: pixel = 255

                     new_image[r, c, ch] = pixel
                     pass
                 pass
             pass
         pass
    elif type == '-':
         for ch in range(chs):
             for r in range(h):
                 for c in range(w):
                     pixel = int(first_img[r, c, ch]) - int(second_image[r, c, ch])

                     if pixel < 0: pixel = 0

                     new_image[r, c, ch] = pixel
                     pass
                 pass
             pass
         pass
    else: raise "Wrong operation"

    return new_image


"""___________________________ negative _______________________________________"""

def negative(img):
    """
    Apply algebra on 2-images.\n
    Histogram Equalization is a Pixel Operation that enhances the image by changing in the intensity(value) of a pixel.
    :param img: the input image, should (gray | RGB)
    :return: new enhanced image
    """

    # region Check on the parameters
    shape = np.shape(img)
    try:
        # check in the third-dimension isn't correct for images shape
        if shape[2] != 3 & shape[2] != 1:
            mess = """"
        Image must have 3 dimension (h, w, ch).
        ch must
              1 for gray-level image
              3 for RGB image
        """
            raise TypeError(mess)
    except IndexError:
        # in case 2d array
        shape = shape + (1, )
        img = np.reshape(img, shape)
        pass
    # endregion

    # region Code
    h, w, chs = shape
    img = img.copy()
    for ch in range(chs):
        for r in range(h):
            for c in range(w):
                img[r,c,ch] = 255 - img[r,c,ch]
    # endregion
    return img

"""___________________________ brightness _______________________________________"""


def brightness(img, offset):
    """
    Change the intensity of image pixels.\n
    Brightness is a Pixel Operation that enhances the image by changing in the intensity(value) of a pixel.
    :param img: the source image, should (gray | RGB)
    :param offset: how the changing in the brightness of image, positive move to lightness or negative move to darkness
    :return: new enhanced image
    """

    shape = np.shape(img)

    # to write in image
    img = img.copy()
    # region Check on the parameters
    try:
        # check in the third-dimension isn't correct for images shape
        if shape[2] != 3 & shape[2] != 1:
            mess = """"
        Image must have 3 dimension (h, w, ch).
        ch must
              1 for gray-level image
              3 for RGB image
        """
            raise TypeError(mess)
    except IndexError:
        # in case 2d array
        shape = shape + (1, )
        img = np.reshape(img, shape)
        pass
    # endregion

    h, w, chs = shape

    # region Code
    pixel = 0
    for ch in range(chs):
        for r in range(h):
            for c in range(w):
                pixel = img[r, c, ch] + offset

                # convert into scale
                if pixel > 255: pixel = 255
                if pixel < 0: pixel = 0

                img[r, c, ch] = pixel
                pass
            pass
        pass
    # endregion

    return img

"""___________________________ contrast _______________________________________"""


def contrast(img, new_min: int, new_max: int):
    """
    Contrast change scale of image histogram. i.e, make scaling of image.\n
    Contrast is a Pixel Operation that enhances the image by changing in the intensity(value) of a pixel.
    :param img: the source image, should (gray | RGB)
    :param new_min: minimum pixel in new range
    :param new_max: maximum pixel in new range
    :return: new enhanced image
    """

    shape = np.shape(img)

    # region Check on the parameters
    try:
        # check in the third-dimension isn't correct for images shape
        if shape[2] != 3 & shape[2] != 1:
            mess = """"
        Image must have 3 dimension (h, w, ch).
        ch must
              1 for gray-level image
              3 for RGB image
        """
            raise TypeError(mess)
    except IndexError:
        # in case 2d array
        shape = shape + (1, )
        img = np.reshape(img, shape)
        pass
    # endregion

    # region Code
    h , w, ch = np.shape(img)
    new_img = np.zeros((h , w, ch), np.uint8)
    for i in range(ch):
        # get min & max from image
        min = np.min(img[:,:,i])
        max = np.max(img[:,:,i])

        for r in range(h):
            for c in range(w):
                # calculate corresponding pixel value
                pixel = (img[r,c,i] - min) / (max - min) * (new_max - new_min) + new_min

                # convert into scale
                if pixel > 255: pixel = 255
                if pixel < 0: pixel = 0

                new_img[r,c,i] = pixel
                pass
            pass
        pass
    # endregion

    return  new_img


"""___________________________convert_to_gray_______________________________________"""


def gray_simple(img, chanel):
    """
     Convert to RGB image to gray scale
    :param img: the source image, should RGB
    :param chanel: what channel, to keep it [0 | 1 | 2]
    :return: new enhanced image
    """

    # region Check on the parameters
    # check in the third-dimension isn't correct for images shape
    shape = np.shape(img)
    if shape[2] != 3 :
        mess = """"
        Image must have 3 dimension (h, w, ch).
        ch must
              3 for RGB image
        """
        raise TypeError(mess)
    # endregion

    return img[:,:, chanel]
def gray_avg(img):
    """
     Convert to RGB image to gray scale
    :param img: the source image, should RGB
    :return: new enhanced image
    """

    # region Check on the parameters
    # check in the third-dimension isn't correct for images shape
    shape = np.shape(img)
    if shape[2] != 3 :
        mess = """"
        Image must have 3 dimension (h, w, ch).
        ch must
              3 for RGB image
        """
        raise TypeError(mess)
    # endregion

    return ((img[:, :, 0] + img[:, :, 1] + img[:, :, 2] ) / 3).astype(np.uint8)
def gray_lum(img, alpha, beta, gamma):
    """
     Convert to RGB image to gray scale
    :param img: the source image, should RGB
    :param alpha: red chanel correlation
    :param beta: green chanel correlation
    :param gamma: blue chanel correlation
    :return: new enhanced image
    """

    # region Check on the parameters
    # check in the third-dimension isn't correct for images shape
    shape = np.shape(img)
    if shape[2] != 3 :
        mess = """"
        Image must have 3 dimension (h, w, ch).
        ch must
              3 for RGB image
        """
        raise TypeError(mess)
    # endregion

    return ((img[:, :, 0] * alpha) + (img[:, :, 1] * beta) + (img[:, :, 2] * gamma)).astype(np.uint8)
def gray_dest(img):
    """
     Convert to RGB image to gray scale
    :param img: the source image, should RGB
    :return: new enhanced image
    """

    # region Check on the parameters
    # check in the third-dimension isn't correct for images shape
    shape = np.shape(img)
    if shape[2] != 3 :
        mess = """"
        Image must have 3 dimension (h, w, ch).
        ch must
              3 for RGB image
        """
        raise TypeError(mess)
    # endregion

    return ((np.maximum(img[:, :, 0], np.maximum(img[:, :, 1], img[:, :, 2])) + \
             np.minimum(img[:, :, 0], np.minimum(img[:, :, 1], img[:, :, 2]))) / 2).astype(np.uint8)
def gray_max(img):
    """
     Convert to RGB image to gray scale
    :param img: the source image, should RGB
    :return: new enhanced image
    """

    # region Check on the parameters
    # check in the third-dimension isn't correct for images shape
    shape = np.shape(img)
    if shape[2] != 3 :
        mess = """"
        Image must have 3 dimension (h, w, ch).
        ch must
              3 for RGB image
        """
        raise TypeError(mess)
    # endregion

    return (np.maximum(img[:, :, 0], np.maximum(img[:, :, 1], img[:, :, 2]))).astype(np.uint8)
def gray_min(img):
    """
     Convert to RGB image to gray scale
    :param img: the source image, should RGB
    :return: new enhanced image
    """

    # region Check on the parameters
    # check in the third-dimension isn't correct for images shape
    shape = np.shape(img)
    if shape[2] != 3 :
        mess = """"
        Image must have 3 dimension (h, w, ch).
        ch must
              3 for RGB image
        """
        raise TypeError(mess)
    # endregion

    return (np.minimum(img[:, :, 0], np.minimum(img[:, :, 1], img[:, :, 2]))).astype(np.uint8)


"""__________________________histogram matching function_____________________________ """


def histogram_equalization(img):
    """
    Histogram Equalization change image histogram distribution into uniform distribution.

     Histogram Equalization is a Pixel Operation that enhances the image by changing in the intensity(value) of a pixel.
    :param img: the source image, should (gray | RGB)
    :return: unique pixels and their new values.
    """

    # histogram equalization
    values, counts = np.unique(img, return_counts=True)
    cms = np.cumsum(counts); new_pixels = np.round((cms / cms[-1]) * values)

    return values, new_pixels

def matching(src_img, targ_img):
    """
    Histogram Matching change image histogram distribution as the target histogram distribution.\n
    Histogram Equalization is a Pixel Operation that enhances the image by changing in the intensity(value) of a pixel.
    :param src_img: the source image, should (gray | RGB)
    :param targ_img: the target image, should (gray | RGB)
    :return: new enhanced image
    """

    # region Check on the parameters
    src_img = src_img.copy()
    shape_src = np.shape(src_img)
    shape_targ = np.shape(targ_img)
    try:
        # check in the third-dimension isn't correct for images shape
        if (shape_src[2] != 3 & shape_src[2] != 1) | (shape_targ[2] != 3 & shape_targ[2] != 1):
            mess = """"
        Image must have 3 dimension (h, w, ch).
        ch must
              1 for gray-level image
              3 for RGB image
        """
            raise TypeError(mess)
    except IndexError:
        # in case 2d array
        shape_src = shape_src + (1, )
        src_img = np.reshape(src_img, shape_src)

        shape_targ = shape_targ + (1, )
        targ_img = np.reshape(targ_img, shape_targ)
        pass
    # endregion

    # region Code
    def get_nearest(source_values, target_values):
        s_len, t_len = len(source_values), len(target_values)
        indexes = []
        for s_v in range(s_len):
            min_diff, min_index = 300, 300
            # get the nearest value to the source values
            for t_v in range(t_len):
                diff = abs(source_values[s_v] - target_values[t_v])
                if diff == 0:
                    min_index = t_v; break
                if diff < min_diff: min_diff, min_index = diff, t_v
                pass
            indexes.append(min_index)
            pass

        return indexes

    src_hist, targ_hist = histogram_equalization(src_img), histogram_equalization(targ_img)
    src_values, src_pixels = src_hist; targ_values, targ_pixels = targ_hist
    nearest_pxs = get_nearest(src_pixels, targ_pixels)

    colors = targ_values[nearest_pxs]
    # apply on image
    for i in range(len(src_values)): src_img[np.where(src_img == src_values[i])] = colors[i]
    # endregion

    return src_img


"""function for adding watermark image to origenal image (using bit_plane slicing)"""

distance = lambda pixel, centre: np.sqrt(((pixel[0] - centre[0])** 2) + ((pixel[1] - centre[1])** 2) )


def lowpass(img, type_pass:int, threshold:int , n= None):
    """
    Low pass is used to make smoothing.
    :param img: Input image, should be gray image
    :param type_pass: type of filter 0 (ideal) | 1 (butterworth) | 2(guassiam)
    :param threshold: the value that determine, which pixels are passed
    :param n: for butterworth
    :return: new enhanced image
    """

    # region Check on the parameters
    img = img.copy()
    # check in the third-dimension isn't correct for images shape
    try:
        if np.shape(img)[2] != 1:
            img = gray_avg(img)
            img = np.reshape(img, np.shape(img)[:2])
    except IndexError:pass
    # endregion

    for_img = np.fft.fftshift(np.fft.fft2(img))

    py.imshow(np.log1p(np.abs(for_img)), cmap='gray')
    py.title("Image in Frequency Domain"); py.show()
    shape = np.shape(img)
    h, w = shape
    cen_pix = (h / 2, w / 2)
    img_mat = np.zeros(shape, dtype=np.float32)

    if type_pass == 0:
        for i in range(h):
            for j in range(w):
                if distance((i,j), cen_pix) <= threshold: img_mat[i, j] = 1
        pass
    elif type_pass == 1:
        for i in range(h):
            for j in range(w):
                img_mat[i, j] = 1 / (1 + (distance((i,j), cen_pix)/ threshold)**n)
        pass
    elif type_pass == 2:
        for i in range(h):
            for j in range(w):
                img_mat[i, j] = np.exp((-1 * (distance((i,j), cen_pix)**2)) / (2* (threshold**2)))
        pass


    new_img = for_img * img_mat
    py.imshow(np.log1p(np.abs(new_img)), cmap='gray')
    py.title("Mask Image in Frequency Domain"); py.show()

    return np.abs(np.fft.ifft2(np.fft.ifftshift(new_img)))


def highpass(img, type_pass:int, threshold:int , n= None):
    """
    High pass is used to make edge detection
    :param img: Input image, should be gray image
    :param type_pass: type of filter
    :param threshold: the value that determine, which pixels are passed
    :param n: for dfd
    :return: new enhanced image
    """

    # region Check on the parameters
    img = img.copy()
    # check in the third-dimension isn't correct for images shape
    try:
        if np.shape(img)[2] != 1:
            img = gray_avg(img)
            img = np.reshape(img, np.shape(img)[:2])
    except IndexError:
        pass
    # endregion

    for_img = np.fft.fftshift(np.fft.fft2(img))

    py.imshow(np.log1p(np.abs(for_img)), cmap='gray')
    py.title("Image in Frequency Domain"); py.show()
    shape = np.shape(img)
    h, w = shape
    cen_pix = (h / 2, w / 2)
    img_mat = np.zeros(shape, dtype=np.float32)

    if type_pass == 0:
        for i in range(h):
            for j in range(w):
                if distance((i,j), cen_pix) > threshold: img_mat[i, j] = 1
        pass
    elif type_pass == 1:
        for i in range(h):
            for j in range(w):
                img_mat[i, j] = 1 / (1 + (threshold / distance((i,j), cen_pix))**n)
        pass
    elif type_pass == 2:
        for i in range(h):
            for j in range(w):
                img_mat[i, j] = 1 - np.exp((-1 * (distance((i,j), cen_pix)**2)) / (2* (threshold**2)))
        pass


    new_img = for_img * img_mat
    py.imshow(np.log1p(np.abs(new_img)), cmap='gray')
    py.title("Mask Image in Frequency Domain"); py.show()

    return np.abs(np.fft.ifft2(np.fft.ifftshift(new_img)))

"""_____________________________ideal low pass filter__________________________________ """

def ideal_lpf(d0, img):
    return lowpass(img, type_pass=0, threshold=d0)

"""_______________________________gaussian low pass filter_______________________ """


def gaussian_lpf(d0, img):
    return lowpass(img, type_pass=2, threshold=d0)


"""_________________________butterworth low pass filter_______________________________"""


def butterworth_lpf(d0, n, img):
    return lowpass(img, type_pass=1, threshold=d0, n=n)


"""____________________________________ideal high pass filter________________________ """


def ideal_hpf(d0, img):
    return highpass(img, type_pass=0, threshold=d0)


"""_______________________________gaussian high pass filter_________________________ """


def gaussian_hpf(d0, img):
    return highpass(img, type_pass=2, threshold=d0)


"""_____________________________butterworth high pass filter______________________ """


def butterworth_hpf(d0, n, img):
    return highpass(img, type_pass=1, threshold=d0, n=n)


# """____________________________________horizontal_Filter______________________________"""
def edge_detection(img, type_edge: int, type_padding = False):
    """
    Edge Detection is useful in Segmentation.
    Edge Detection is Neighbor Operation that enhance image by change the pixel value based on its adjacent (neighbor) pixels.
    :param img: the input image, should gray
    :param type_edge: type of edge detection filter, 0 (Laplacian) | 1 (Prewitt x-axis) | 2 (Prewitt y-axis)
    :param type_padding: type of padding add zeros (False) or replicating the first and last row & column (True).
    :return: new enhanced image
    """

    img = img.copy()
   # try:
    if np.shape(img)[2] != 1: img = gray_avg(img)
    # except IndexError:
    #     img = np.reshape(img, (np.shape(img), 1))

    if type_edge == 0: return contrast(aplay_filter(img, np.array([
        [
           0, 1, 0
        ],
        [
            1, -4, 1
        ],
        [
            0, 1, 0
        ]
    ]), type_padding), 0, 255)
    elif type_edge == 1: return contrast(aplay_filter(img, np.array([
        [
           -1, 0, 1
        ],
        [
            -1, 0, 1
        ],
        [
            -1, 0, 1
        ]
    ]), type_padding), 0, 255)
    elif type_edge == 2: return contrast(aplay_filter(img, np.array([
        [
           -1, -1, -1
        ],
        [
            0, 0, 0
        ],
        [
            1, 1, 1
        ]
    ]), type_padding), 0, 255)
    else: raise TypeError("type_smoothing Must 0 | 1 | 2")

def edg_horizontal_Filter(img):
    return edge_detection(img, type_edge=1)

"""______________________________________vertical_Filter_______________________"""


def edg_vertical_Filter(img):
    return edge_detection(img, type_edge=2)


"""________________________________diagonal\_Filter_____________________________"""


def edg_diagonal_r_Filter(img):pass
    # ori_img = convert_to_gray(img)
    # # create mask
    # mask = np.array([[0, 1, 2], [-2, 0, 1], [-2, -1, 0]])
    # [rows, cols] = ori_img.shape
    # new_img = np.zeros((rows, cols), dtype=ori_img.dtype)
    # mask2 = mask * 0  # overlapped regionS
    # # create padding
    # mk = mask.shape
    # top = bottom = np.int32((mk[0] - 1) / 2)
    # right = left = np.int32((mk[1] - 1) / 2)
    # # adding pad to image
    # pad_img = cv2.copyMakeBorder(ori_img, top, bottom, left, right, cv2.BORDER_REFLECT)
    # rows = pad_img.shape[0]
    # cols = pad_img.shape[1]
    # # horizontal detection Filter.
    # # top=numbur of pad
    # for i in range(top, rows - top):
    #     for j in range(top, cols - top):
    #         # take the overlapped region from the image
    #         mi = 0
    #         mj = 0
    #         for mr in range((i - top), (i + top + 1)):
    #             for mc in range((j - top), (j + top + 1)):
    #                 mask2[mi, mj] = int(pad_img[mr, mc])
    #                 mj = mj + 1
    #             mj = 0
    #             mi = mi + 1
    #         # put new value in the pixel
    #         new_pixel = np.sum(mask2 * mask)
    #         new_img[i - top, j - top] = new_pixel
    # return new_img

#
# """________________________________________diagonal/_Filter_________________________"""
#
#
def edg_diagonal_l_Filter(img):pass
#     ori_img = convert_to_gray(img)
#     # create mask
#     mask = np.array([[2, 1, 0], [1, 0, -1], [0, -1, -2]])
#     [rows, cols] = ori_img.shape
#     new_img = np.zeros((rows, cols), dtype=ori_img.dtype)
#     mask2 = mask * 0  # overlapped regionS
#     # create padding
#     mk = mask.shape
#     top = bottom = np.int32((mk[0] - 1) / 2)
#     right = left = np.int32((mk[1] - 1) / 2)
#     # adding pad to image
#     pad_img = cv2.copyMakeBorder(ori_img, top, bottom, left, right, cv2.BORDER_REFLECT)
#     rows = pad_img.shape[0]
#     cols = pad_img.shape[1]
#     # horizontal detection Filter.
#     # top=numbur of pad
#     for i in range(top, rows - top):
#         for j in range(top, cols - top):
#             # take the overlapped region from the image
#             mi = 0
#             mj = 0
#             for mr in range((i - top), (i + top + 1)):
#                 for mc in range((j - top), (j + top + 1)):
#                     mask2[mi, mj] = int(pad_img[mr, mc])
#                     mj = mj + 1
#                 mj = 0
#                 mi = mi + 1
#             # put new value in the pixel
#             new_pixel = np.sum(mask2 * mask)
#             new_img[i - top, j - top] = new_pixel
#     return new_img

#
# """_________________________________ point edg detiction filters__________________"""
#
#
def edg_point_Filter(mask_size, ori_img):pass
#     # create mask
#     mask = np.ones((mask_size, mask_size), dtype=int) * -1
#     mid = int(mask.__len__() / 2)
#     mask[mid, mid] = (mask_size * mask_size) - 1
#     mask2 = mask * 0  # overlapped regionS
#     [rows, cols, ch] = ori_img.shape
#     sharpening_img = np.zeros((rows, cols, ch), dtype=ori_img.dtype)
#     # create padding
#     mk = mask.shape
#     top = bottom = np.int32((mk[0] - 1) / 2)
#     right = left = np.int32((mk[1] - 1) / 2)
#     # adding pad to image
#     pad_img = cv2.copyMakeBorder(ori_img, top, bottom, left, right, cv2.BORDER_REFLECT)
#     rows = pad_img.shape[0]
#     cols = pad_img.shape[1]
#     ch = pad_img.shape[2]
#     # Sharpening filter
#     # top=numbur of pad
#     for k in range(ch):
#         for i in range(top, rows - top):
#             for j in range(top, cols - top):
#                 # take the overlapped region from the image
#                 mi = 0
#                 mj = 0
#                 for mr in range((i - top), (i + top + 1)):
#                     for mc in range((j - top), (j + top + 1)):
#                         mask2[mi, mj] = int(pad_img[mr, mc, k])
#                         mj = mj + 1
#                     mj = 0
#                     mi = mi + 1
#                 # put new value in the pixel
#                 new_pixel = np.sum(mask2 * mask)
#                 sharpening_img[i - top, j - top, k] = new_pixel
#     return sharpening_img
#
#
# """______________________________ Sharpening horizontal_Filter______________________"""
#
#
def Sharpening_horizontal_Filter(ori_img):pass
#     # create mask
#     mask = np.array([[0, 1, 0], [0, 1, 0], [0, -1, 0]])
#     [rows, cols, chs] = ori_img.shape
#     new_img = np.zeros((rows, cols, chs), dtype=ori_img.dtype)
#     mask2 = mask * 0  # overlapped regionS
#     # create padding
#     mk = mask.shape
#     top = bottom = np.int32((mk[0] - 1) / 2)
#     right = left = np.int32((mk[1] - 1) / 2)
#     # adding pad to image
#     pad_img = cv2.copyMakeBorder(ori_img, top, bottom, left, right, cv2.BORDER_REFLECT)
#     rows = pad_img.shape[0]
#     cols = pad_img.shape[1]
#     ch = pad_img.shape[2]
#     # Sharpening filter
#     # top=numbur of pad
#     for k in range(ch):
#         for i in range(top, rows - top):
#             for j in range(top, cols - top):
#                 # take the overlapped region from the image
#                 mi = 0
#                 mj = 0
#                 for mr in range((i - top), (i + top + 1)):
#                     for mc in range((j - top), (j + top + 1)):
#                         mask2[mi, mj] = int(pad_img[mr, mc, k])
#                         mj = mj + 1
#                     mj = 0
#                     mi = mi + 1
#                 # put new value in the pixel
#                 new_pixel = np.sum(mask2 * mask)
#                 new_img[i - top, j - top, k] = new_pixel
#     return new_img
#
#
# """___________________________________Sharpening vertical_Filter__________________"""
#
#
def Sharpening_vertical_Filter(ori_img):pass
#     # create mask
#     mask = np.array([[0, 0, 0], [1, 1, -1], [0, 0, 0]])
#     [rows, cols, chs] = ori_img.shape
#     new_img = np.zeros((rows, cols, chs), dtype=ori_img.dtype)
#     mask2 = mask * 0  # overlapped regionS
#     # create padding
#     mk = mask.shape
#     top = bottom = np.int32((mk[0] - 1) / 2)
#     right = left = np.int32((mk[1] - 1) / 2)
#     # adding pad to image
#     pad_img = cv2.copyMakeBorder(ori_img, top, bottom, left, right, cv2.BORDER_REFLECT)
#     rows = pad_img.shape[0]
#     cols = pad_img.shape[1]
#     ch = pad_img.shape[2]
#     # Sharpening filter
#     # top=numbur of pad
#     for k in range(ch):
#         for i in range(top, rows - top):
#             for j in range(top, cols - top):
#                 # take the overlapped region from the image
#                 mi = 0
#                 mj = 0
#                 for mr in range((i - top), (i + top + 1)):
#                     for mc in range((j - top), (j + top + 1)):
#                         mask2[mi, mj] = int(pad_img[mr, mc, k])
#                         mj = mj + 1
#                     mj = 0
#                     mi = mi + 1
#                 # put new value in the pixel
#                 new_pixel = np.sum(mask2 * mask)
#                 new_img[i - top, j - top, k] = new_pixel
#     return new_img
#
#
# """_______________________________Sharpening diagonal_r_Filter__________________________"""
#
#
def Sharpening_diagonal_r_Filter(ori_img):pass
#     # create mask
#     mask = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]])
#     [rows, cols, chs] = ori_img.shape
#     new_img = np.zeros((rows, cols, chs), dtype=ori_img.dtype)
#     mask2 = mask * 0  # overlapped regionS
#     # create padding
#     mk = mask.shape
#     top = bottom = np.int32((mk[0] - 1) / 2)
#     right = left = np.int32((mk[1] - 1) / 2)
#     # adding pad to image
#     pad_img = cv2.copyMakeBorder(ori_img, top, bottom, left, right, cv2.BORDER_REFLECT)
#     rows = pad_img.shape[0]
#     cols = pad_img.shape[1]
#     ch = pad_img.shape[2]
#     # Sharpening filter
#     # top=numbur of pad
#     for k in range(ch):
#         for i in range(top, rows - top):
#             for j in range(top, cols - top):
#                 # take the overlapped region from the image
#                 mi = 0
#                 mj = 0
#                 for mr in range((i - top), (i + top + 1)):
#                     for mc in range((j - top), (j + top + 1)):
#                         mask2[mi, mj] = int(pad_img[mr, mc, k])
#                         mj = mj + 1
#                     mj = 0
#                     mi = mi + 1
#                 # put new value in the pixel
#                 new_pixel = np.sum(mask2 * mask)
#                 new_img[i - top, j - top, k] = new_pixel
#     return new_img
#
#
# """_____________________________Sharpening diagonal_l_Filter______________________"""
#
#
def Sharpening_diagonal_l_Filter(ori_img):pass
#     # create mask
#     mask = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
#     [rows, cols, chs] = ori_img.shape
#     new_img = np.zeros((rows, cols, chs), dtype=ori_img.dtype)
#     mask2 = mask * 0  # overlapped regionS
#     # create padding
#     mk = mask.shape
#     top = bottom = np.int32((mk[0] - 1) / 2)
#     right = left = np.int32((mk[1] - 1) / 2)
#     # adding pad to image
#     pad_img = cv2.copyMakeBorder(ori_img, top, bottom, left, right, cv2.BORDER_REFLECT)
#     rows = pad_img.shape[0]
#     cols = pad_img.shape[1]
#     ch = pad_img.shape[2]
#     # Sharpening filter
#     # top=numbur of pad
#     for k in range(ch):
#         for i in range(top, rows - top):
#             for j in range(top, cols - top):
#                 # take the overlapped region from the image
#                 mi = 0
#                 mj = 0
#                 for mr in range((i - top), (i + top + 1)):
#                     for mc in range((j - top), (j + top + 1)):
#                         mask2[mi, mj] = int(pad_img[mr, mc, k])
#                         mj = mj + 1
#                     mj = 0
#                     mi = mi + 1
#                 # put new value in the pixel
#                 new_pixel = np.sum(mask2 * mask)
#                 new_img[i - top, j - top, k] = new_pixel
#     return new_img
#
#
# """________________________________Sharpening filters____________________________"""
#
#
def sharpening_Filter(mask_size, ori_img):pass
#     # create mask
#     mask = np.ones((mask_size, mask_size), dtype=int) * -1
#     mid = int(mask.__len__() / 2)
#     mask[mid, mid] = mask_size * mask_size
#     mask2 = mask * 0  # overlapped regionS
#     [rows, cols, ch] = ori_img.shape
#     sharpening_img = np.zeros((rows, cols, ch), dtype=ori_img.dtype)
#     # create padding
#     mk = mask.shape
#     top = bottom = np.int32((mk[0] - 1) / 2)
#     right = left = np.int32((mk[1] - 1) / 2)
#     # adding pad to image
#     pad_img = cv2.copyMakeBorder(ori_img, top, bottom, left, right, cv2.BORDER_REFLECT)
#     rows = pad_img.shape[0]
#     cols = pad_img.shape[1]
#     ch = pad_img.shape[2]
#     # Sharpening filter
#     # top=numbur of pad
#     for k in range(ch):
#         for i in range(top, rows - top):
#             for j in range(top, cols - top):
#                 # take the overlapped region from the image
#                 mi = 0
#                 mj = 0
#                 for mr in range((i - top), (i + top + 1)):
#                     for mc in range((j - top), (j + top + 1)):
#                         mask2[mi, mj] = int(pad_img[mr, mc, k])
#                         mj = mj + 1
#                     mj = 0
#                     mi = mi + 1
#                 # put new value in the pixel
#                 new_pixel = np.sum(mask2 * mask)
#                 sharpening_img[i - top, j - top, k] = new_pixel
#     return sharpening_img
#
#
# """___________________________histogram equalization function_________________________________"""
#
#
def equalization(img):
    """
    Apply hist_equalization on image
    :param img: the source image, should (gray | RGB)
    :return: new enhanced image
    """

    # to write in image
    img = img.copy()
    # region Check on the parameters
    shape = np.shape(img)
    try:
        # check in the third-dimension isn't correct for images shape
        if shape[2] != 3 & shape[2] != 1:
            mess = """"
        Image must have 3 dimension (h, w, ch).
        ch must
              1 for gray-level image
              3 for RGB image
        """
            raise TypeError(mess)
    except IndexError:
        # in case 2d array
        shape = shape + (1, )
        img = np.reshape(img, shape)
        pass
    # endregion

    # region Code

    # histogram equalization
    values, new_pixels = histogram_equalization(image)

    # apply on image
    for i in range(len(values)): img[np.where(img == values[i])] = new_pixels[i]
    # endregion

    return img

def power_law(img, gamma):
    """
    Power-Law Transform change image histogram distribution into normal distribution.

    Power-Law Transform is a Pixel Operation that enhances the image by changing in the intensity(value) of a pixel.
    :param img: the source image, should (gray | RGB)
    :param gamma:
    :return: new enhanced image
    """

    shape = np.shape(img)

    # region Check on the parameters
    try:
        # check in the third-dimension isn't correct for images shape
        if shape[2] != 3 & shape[2] != 1:
            mess = """"
        Image must have 3 dimension (h, w, ch).
        ch must
              1 for gray-level image
              3 for RGB image
        """
            raise TypeError(mess)
    except IndexError:
        # in case 2d array
        shape = shape + (1, )
        img = np.reshape(img, shape)
        pass
    # endregion

    new_img = (np.power(img, gamma)).astype(np.uint8)

    return contrast(new_img,0,255)

def minimize(img, reduc):pass
#     new_rows = int(img.shape[0] / reduc)
#     new_cols = int(img.shape[1] / reduc)
#     ch = img.shape[2]
#     imagezoomout = np.zeros((new_rows, new_cols, ch), dtype=img.dtype)
#     for k in range(ch):
#         for i in range(new_rows):
#             for j in range(new_cols):
#                 imagezoomout[i, j, k] = img[i * reduc, j * reduc, k]
#     return imagezoomout
#
#

"""___________________________one order maximize_______________________________________"""

def direct_mapp(img, factor: int, type_: bool):
    """
    Apply Direct Map 0-order to resize the image, and change width and height with the same factor.

    Direct Map is a Geometry Operation that enhances the image by changing the location of a pixel.
    **Zooming in, only**
    :param img: the source image, should (gray | RGB)
    :param factor: the extent to which the image is zoomed from the input image
    :param type_: 0-order | 1-order
    :return: new resized (enhanced) image
    """

    shape = np.shape(img)

    # region Check on the parameters
    # check on factor for zooming in
    if factor < 1: raise ValueError("Factor must bigger than 1")
    try:
        # check in the third-dimension isn't correct for images shape
        if shape[2] != 3 & shape[2] != 1:
            mess = """"
        Image must have 3 dimension (h, w, ch).
        ch must
              1 for gray-level image
              3 for RGB image
        """
            raise TypeError(mess)
    except IndexError:
        # in case 2d array
        shape = shape + (1, )
        img = np.reshape(img, shape)
        pass
    # endregion

    h, w, chs = shape
    n_h, n_w = int(h * factor), int(w * factor)
    new_img = np.zeros((n_h, n_w, chs), np.uint8)

    # region Code
    def zero_order():
        """
        just repeat old-pixels, in the kernel "shape(factor, factor)".
        """
        # assign to new image
        for ch in range(chs):
            for r in range(h):
                for c in range(w):
                    new_img[r * factor: (r + 1) * factor,  c * factor: (c + 1) * factor, ch] = img[r, c, ch]
                    pass
                pass
            pass
        pass
    def one_order():
        """
        apply this equation on image:
        new_image = round(((max_pixel - min_pixel) / factor) * i) + min_pixel)
        """

        step, ind_min, min_pi = 0,0,0
        for ch in range(chs):
            # O(h * (w * factor))
            # fill pixels of main rows
            for r in range(h):
                # fil the main pixels
                # i.e, assign the pixels from old image into the new position in new image
                cur_row = r * factor
                new_img[cur_row, 0, ch] = img[r, 0, ch]

                # fil the pixels between the main pixels
                for c in range(1, w):
                    pi_pr, pi_cur = (c - 1) * factor,  c * factor
                    # fil the main pixels
                    new_img[cur_row, pi_cur, ch] = img[r, c, ch]

                    # get the minimum pixel in current row between adjacent two-pixels.
                    if new_img[cur_row, pi_pr , ch] < new_img[cur_row, pi_cur, ch]:
                        min_pi , ind_min = new_img[cur_row, pi_pr , ch], pi_pr
                        step = (new_img[cur_row, pi_cur , ch] - min_pi) / factor
                        pass
                    else:
                        min_pi , ind_min = new_img[cur_row, pi_cur , ch], pi_cur
                        step = (new_img[cur_row, pi_pr , ch] - min_pi) / factor
                        pass

                    # fill pixels between adjacent two-pixels
                    for p in range(pi_pr+ 1, pi_cur):
                        new_img[cur_row, p, ch] = round((step * abs(p - ind_min)) + min_pi)
                        pass
                pass

                # fill remained pixels, after the last pixel in the row in original image
                o = (w - 1) * factor
                new_img[cur_row,  o:, ch] = new_img[cur_row, o, ch]
                pass

            step, ind_min, min_pi = 0, 0, 0
            # O((h * factor) * n_w)
            # fill pixels of remained rows
            for c in range(n_w):
                for r in range((h - 1) * factor):
                    # skip main rows in assigning operation
                    if r % factor == 0:
                        # get the minimum pixel in current column between adjacent two-pixels.
                        pi_cur, pi_nxt = r,  r + factor
                        if new_img[pi_cur, c, ch] < new_img[pi_nxt, c, ch]:
                            min_pi , ind_min = new_img[pi_cur, c , ch], pi_cur
                            step = (new_img[pi_nxt, c , ch] - min_pi) / factor
                            pass
                        else:
                           min_pi , ind_min = new_img[pi_nxt, c , ch], pi_nxt
                           step = (new_img[pi_cur, c, ch] - min_pi) / factor
                           pass
                    else:
                        # fill pixels between adjacent two-pixels
                        new_img[r, c, ch] = round((step * abs(r - ind_min)) + min_pi); pass
                    pass

                # fill remained pixels, after the last pixel in the row in original image
                o = (h - 1) * factor
                new_img[o + 1: ,  c, ch] =  new_img[o,  c, ch]
                pass

            pass

        pass
    # endregion

    if type_: one_order()
    else: zero_order()

    return new_img
def order1_max(img, fact):
    return direct_mapp(img, fact, True)

def median_Filter(mask_size, ori_img):pass

#     # create mask
#     mask = np.ones((mask_size, mask_size), dtype=int)
#     [rows, cols, ch] = ori_img.shape
#     median_img = np.zeros((rows, cols, ch), dtype=ori_img.dtype)
#     mask2 = mask * 0  # overlapped regionS
#     # create padding
#     mk = mask.shape
#     top = bottom = np.int32((mk[0] - 1) / 2)
#     right = left = np.int32((mk[1] - 1) / 2)
#     # adding pad to image
#     pad_img = cv2.copyMakeBorder(ori_img, top, bottom, left, right, cv2.BORDER_REFLECT)
#     rows = pad_img.shape[0]
#     cols = pad_img.shape[1]
#     ch = pad_img.shape[2]
#     # Smoothing with Median Filter.
#     # top=numbur of pad
#     for k in range(ch):
#         for i in range(top, rows - top):
#             for j in range(top, cols - top):
#                 # take the overlapped region from the image
#                 mi = 0
#                 mj = 0
#                 for mr in range((i - top), (i + top + 1)):
#                     for mc in range((j - top), (j + top + 1)):
#                         mask2[mi, mj] = int(pad_img[mr, mc, k])
#                         mj = mj + 1
#                     mj = 0
#                     mi = mi + 1
#                 # put new value in the pixel
#                 temp = np.ravel(mask2)
#                 temp.sort()
#                 new_pixel = temp[int(len(temp) / 2)]
#                 median_img[i - top, j - top, k] = new_pixel
#     return median_img

"""___________________________select image_______________________________________"""


def select_image():
    # grab a reference to the image panels
    global panelA, panelB, image, edged
    # open a file chooser dialog and allow the user to select an input
    # image
    path = tkFileDialog.askopenfilename(filetypes=(("jpg files", "*.jpg"),
                                                   ("webp", "*.webp"), ("png", "*.png"), ("jpeg", "*.jpeg"),
                                                   ("pjp", "*.pjp")))
    # ensure a file path was selected
    if len(path) > 0:

        # load the image from disk, convert it to grayscale, and detect
        # edges in it

        image = cv2.cvtColor(cv2.imread(path, cv2.COLOR_BGR2RGB), cv2.COLOR_BGR2RGB)
        edged = mask(mas_k, image)

        # convert the images to PIL format...
        image = Image.fromarray(cv2.resize(image, (300, 250), interpolation=cv2.INTER_AREA))
        edged2 = Image.fromarray(cv2.resize(edged, (300, 250), interpolation=cv2.INTER_AREA))
        # ...and then to ImageTk format
        image = ImageTk.PhotoImage(image)
        edged2 = ImageTk.PhotoImage(edged2)
        # if the panels are None, initialize them
        if panelA is None or panelB is None:
            # the first panel will store our original image
            panelA = Label(image=image)
            panelA.image = image
            panelA.place(x=350, y=400)
            # while the second panel will store the edge map
            panelB = Label(image=edged2)
            panelB.image = edged2
            panelB.place(x=700, y=400)
        # otherwise, update the image panels
        else:
            # update the pannels
            panelA.configure(image=image)
            panelB.configure(image=edged2)
            panelA.image = image
            panelB.image = edged2


"""___________________________select image1_______________________________________"""


def selectimage1():
    # grab a reference to the image panels
    global panelC, image1
    # open a file chooser dialog and allow the user to select an input
    # image
    path = tkFileDialog.askopenfilename(filetypes=(("jpg files", "*.jpg"),
                                                   ("webp", "*.webp"), ("png", "*.png"), ("jpeg", "*.jpeg"),
                                                   ("pjp", "*.pjp")))
    # ensure a file path was selected
    if len(path) > 0:

        # load the image from disk, convert it to grayscale, and detect
        # edges in it

        image1 = cv2.cvtColor(cv2.imread(path, cv2.COLOR_BGR2RGB), cv2.COLOR_BGR2RGB)
        image2 = image1

        # convert the images to PIL format...
        image2 = Image.fromarray(cv2.resize(image2, (300, 250), interpolation=cv2.INTER_AREA))

        # ...and then to ImageTk format
        image2 = ImageTk.PhotoImage(image2)

        # if the panels are None, initialize them
        if panelC is None:
            # the first panel will store our original image
            panelC = Label(image=image2)
            panelC.image = image2
            panelC.place(x=5, y=400)

        # otherwise, update the image panels
        else:
            # update the pannels
            panelC.configure(image=image2)
            panelC.image = image2


"""_________________________________________________________________________"""


def save_image():
    cv2.imwrite('result.jpg', edged)


"""_________________________________________________________________________"""

# initialize the window toolkit along with the two image panels
root = Tk()
root.geometry("1500x700")
root.title("image processing program")
"""_____________________________________________________________________________"""
# root back ground
img = Image.open('background .webp')
bg = ImageTk.PhotoImage(img)
# Add back ground image
label1 = Label(root, image=bg)
label1.image = bg
label1.place(x=0, y=0)
"""_____________________________________________________________________________"""
# panels to display images
panelA = None
panelB = None
panelC = None
"""_____________________________________________________________________________"""
chf = Label(root, text='choose filter',
            fg="#FFFFFF", bg="#ADD8E6", padx=10, font=("Constantia", 30))
chf.place(x=5, y=10)
# group of filters nees image and size
listbox_need_img_size = Listbox(root,
                                bg="#F0FFF0",
                                font=("Constantia", 20),
                                width=21)
listbox_need_img_size.place(x=5, y=55)
listbox_need_img_size.config(height=5)
for index in range(len(filters_need_img_size)):
    listbox_need_img_size.insert(index, filters_need_img_size[index])

button1 = Button(root, text="Done", fg="#FFFFFF", bg="#ADD8E6", command=choice_need_img_size, font=("Constantia", 20),
                 pady=5)
button1.place(x=5, y=300)

chsp = Label(root, text='Choose Kernal the size', bg="#ADD8E6", font=("Constantia", 25), fg="#FFFFFF")
chsp.place(x=5, y=240)
sp = Spinbox(root, from_=3, to=100, increment=+2, width=30)
sp.place(x=5, y=280)
"""_____________________________________________________________________________"""
chf = Label(root, text='choose filter',
            fg="#FFFFFF", bg="#ADD8E6", padx=10, font=("Constantia", 30))
chf.place(x=350, y=10)
# group of filters nees image and size
listbox_need_img = Listbox(root,
                           bg="#F0FFF0",
                           font=("Constantia", 20),
                           width=24)
listbox_need_img.place(x=350, y=55)
listbox_need_img.config(height=5)
for index in range(len(filters_need_img)):
    listbox_need_img.insert(index, filters_need_img[index])
button2 = Button(root, text="Done", fg="#FFFFFF", bg="#ADD8E6", command=choice_need_img, font=("Constantia", 20),
                 pady=5)
button2.place(x=350, y=250)
"""_____________________________________________________________________________"""

chf = Label(root, text='choose filter',
            fg="#FFFFFF", bg="#ADD8E6", padx=10, font=("Constantia", 30))
chf.place(x=740, y=10)
# group of filters nees image and size
listbox_two_img = Listbox(root,
                          bg="#F0FFF0",
                          font=("Constantia", 20),
                          width=20)
listbox_two_img.place(x=740, y=55)
listbox_two_img.config(height=5)
for index in range(len(filters_two_img)):
    listbox_two_img.insert(index, filters_two_img[index])

button3 = Button(root, text="Done", fg="#FFFFFF", bg="#ADD8E6", command=choice_two_img, font=("Constantia", 20), pady=5)
button3.place(x=740, y=250)

"""_____________________________________________________________________________"""

chf = Label(root, text='choose filter',
            fg="#FFFFFF", bg="#ADD8E6", padx=10, font=("Constantia", 30))
chf.place(x=1070, y=10)
# group of filters nees image and size
listbox_need_img_size_number = Listbox(root,
                                       bg="#F0FFF0",
                                       font=("Constantia", 20),
                                       width=18)
listbox_need_img_size_number.place(x=1070, y=55)
listbox_need_img_size_number.config(height=5)
for index in range(len(filters_need_img_size_number)):
    listbox_need_img_size_number.insert(index, filters_need_img_size_number[index])

button4 = Button(root, text="Done", fg="#FFFFFF", bg="#ADD8E6", command=choice_need_img_size_number,
                 font=("Constantia", 20), pady=5)
button4.place(x=1070, y=360)
chsp = Label(root, text='Choose the size', bg="#ADD8E6", font=("Constantia", 25), fg="#FFFFFF")
chsp.place(x=1070, y=240)
sp2 = Spinbox(root, from_=3, to=100, increment=+2, width=30)
sp2.place(x=1070, y=280)
chsp = Label(root, text='Choose the number', bg="#ADD8E6", font=("Constantia", 22), fg="#FFFFFF")
chsp.place(x=1070, y=300)
sp3 = Spinbox(root, from_=1, to=100, width=30)
sp3.place(x=1070, y=340)

"""_____________________________________________________________________________"""
button5 = Button(root, text="save new image", fg="#FFFFFF", bg="#ADD8E6", command=save_image, font=("Constantia", 20),
                 pady=5)
button5.place(x=1140, y=620)

# kick off the GUI
root.mainloop()
