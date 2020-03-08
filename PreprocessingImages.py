"""
Created on 19th February, 2020
Goal: Provide functions to preprocess images for pre-processing
Author: Matthias Quinn
Source 1: https://www.analyticsvidhya.com/blog/2019/09/9-powerful-tricks-for-working-image-data-skimage-python/
Source 2: https://www.geeksforgeeks.org/python-data-augmentation/
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
from skimage.io import imread, imshow
from skimage.color import rgb2hsv
from skimage.transform import resize
import matplotlib.pyplot as plt
import cv2

# To read an image from your machine:
image1 = imread("SydneyOperaHouse.jfif")
image2 = imread("Number8.jpg")
imshow(image1)
plt.show()
imshow(image2)
plt.show()

"Reading Images from our System using skimage"

# The imread function has a parameter "as_gray" which is used to specify if the image
# is to be converted into a grayscale format.

gray_image = imread("SydneyOperaHouse.jfif", as_gray=True)
imshow(gray_image)
plt.show()

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
imshow(HSVImage)
plt.show()

"Resizing Images:"


# It is generally useful to make sure that each image is the same size
# when they are used as inputs to our model.

# The image function to use is called "resize"
# Custom function below
def resizeImage(image, output_shape, Shape=False):
    if Shape == True:
        print("Height = {}, Width = {}".format(image.shape[0], image.shape[1]))
    a = resize(image, output_shape=output_shape)
    return a


ResizedImage = resizeImage(image=image2, output_shape=(500, 500), Shape=True)
imshow(ResizedImage)
plt.show()

"Converting Images to Grayscale:"


# This function inputs the original image and outputs to a gray color space.
# From 3 channels to 1
def grayImage(image, display=False):
    gray = cv2.cvtColor(image, code=cv2.COLOR_BGR2GRAY)
    if display == True:
        imshow(gray)
        plt.show()
    else:
        return gray

gray = grayImage(image1)












"Data Augmentation:"

# The process of increasing the amount and diversity of the data.
# It doesn't add outside data, but adds new data using old data
# Techniques:
# Rotation
# Shearing - changes the orientation
# Zooming
# Cropping - select a particular area of an image
# Flipping - horizontal or vertical
# Changing the brightness level
