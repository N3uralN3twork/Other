"""
Created on 19th February, 2020
Goal: Provide functions to preprocess images for pre-processing
Author: Matthias Quinn
Source 1: https://www.analyticsvidhya.com/blog/2019/09/9-powerful-tricks-for-working-image-data-skimage-python/
Source 2: https://www.geeksforgeeks.org/python-data-augmentation/
Source 3: https://math.stackexchange.com/questions/906240/algorithms-to-increase-or-decrease-the-contrast-of-an-image/906280#906280
Source 5: https://stackoverflow.com/questions/39308030/how-do-i-increase-the-contrast-of-an-image-in-python-opencv
Source 6: https://towardsdatascience.com/histogram-equalization-5d1013626e64#:~:text=
Source 7: https://stackoverflow.com/questions/25008458/how-to-apply-clahe-on-rgb-color-images
Source 8: https://docs.opencv.org/master/d5/daf/tutorial_py_histogram_equalization.html
Source 9: https://stackoverflow.com/questions/44650888/resize-an-image-without-distortion-opencv

Notes:
    Binary Image = pixels that can have one of only two colors, usually black or white
        Grayscale images are NOT binary: different shades of gray
    For some odd reason, when I try to read a colored image (RGB) into Python using OpenCV,
    I get a picture that is shaded differently than the original image:
        If my input is a red rose image, it somehow reads it as a blue rose image?
    Interesting.
"""
###############################################################################
###                     1.  Define Working Directory                        ###
###############################################################################
import os
abspath = os.path.abspath("C:/Users/miqui/OneDrive/Python Projects/Images")
os.chdir(abspath)
###############################################################################
###                    2. Import Libraries and Models                       ###
###############################################################################
import matplotlib.pyplot as plt
import cv2

"Function to show images later on:"

# CV2 has an image showing function, but it seems to break the Python console everytime I use it.
def show(image):
    plt.imshow(image, aspect="auto")
    plt.show()

"Reading Images from our System using cv2.imread()"

# To read an image from your machine:
image1 = plt.imread("Rose.jpg", )
image2 = plt.imread("Number8.jpg")
image3 = plt.imread("FuzzyTest.jpg")

# Show example:

show(image1)


###############################################################################
###                    3. Image Transformations                             ###
###############################################################################

# Notice how each image is stored as an matrix of numbers.
# The larger the number, the more intense the pixel is.
image1.shape  # The 3 represents the 3 channels in the image, RGB
# It should be noted that color images are 3 times as large as grey-scale images
# Grayscale images are used when the amount of compute power is low

"Changing the Image Format:"
# Hue = degree on the color wheel
# Saturation = represents how strong a color is, from 0 to 100
# Value = mixture of colors
# Lightness = the shade of the image

# HSV = (hue, saturation, value)
# HSL = (hue, saturation, lightness)

# To convert an image to the HSV format:
HSVImage = rgb2hsv(image1)
show(HSVImage)

"Resizing Images:"

# It is generally useful to make sure that each image is the same size
# when they are used as inputs to our model.

# The image function to use is called "resize"
# Custom function below

def resizeImage(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

ResizedImage = resizeImage(image=image2, width=100, height=100)
show(ResizedImage)

"Converting Images to Grayscale:"

# This function inputs the original image and outputs to a gray color space.
# From 3 channels to 1
def grayImage(image, display=False):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if display == True:
        plt.imshow(gray)
        plt.show()
    else:
        return gray


gray = grayImage(image1)
show(gray)

"Changing Brightness and Contrast:"

# This one's a bit more in-depth than the other one's.
# Essentially:
    # f(x) = alpha(X) + beta
    # alpha: controls the contrast of the image  [0, Inf]
    # beta: controls the brightness of the image [-127, 127]
    # alpha 1 beta 0: no change to image

def contrastBright(image, alpha, beta):
    """alpha 1 beta 0: no change to image"""
    adjusted = cv2.convertScaleAbs(image, alpha = alpha, beta = beta)
    return adjusted

contrastBright(image1, alpha=1.5, beta=0)


"Histogram Equalization:"

# A histogram is used to represent the intensity distribution of an image
# X = tonal scale
# Y = number of pixels in the image
# (X,Y) = number of pixels at that specific brightness level
# Helps to spread out the contrast of an image

# Contrast Limited Adaptive Histogram Equalization (CLAHE):
    # Convert an image into HSV or LAB
    # Divide an image into small tiles (8x8 is default)
    # Histogram equalize each tile
    # Apply bi-linear interpolation


def clahe(image, gridSize):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)  # Convert from
    lab_planes = cv2.split(lab)  # Split into LAB components
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(gridSize, gridSize))
    lab_planes[0] = clahe.apply(lab_planes[0])  # Apply to the first dimension only
    lab = cv2.merge(lab_planes)
    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)  # Convert back to BGR
    return bgr


texture = clahe(image3, 8)
show(image3)
show(texture)


















