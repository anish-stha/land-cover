import cv2
import numpy as np
from skimage.segmentation import felzenszwalb
from skimage.util import img_as_float
import matplotlib.pyplot as plt
from matplotlib import colors


image_show = True


def give_segmentation(image):
    print ("Entered segmentation")
    sk_img = img_as_float(image)
    segments_fz = felzenszwalb(sk_img, scale=100, sigma=0.5, min_size=50)
    n_segments = len(np.unique(segments_fz))
    print(n_segments)
    if image_show:
        cmap = colors.ListedColormap(np.random.rand(n_segments, 3))
        plt.figure()
        plt.imshow(segments_fz, interpolation='none', cmap=cmap)
        plt.tight_layout()
        plt.show()
    return segments_fz

# Hue separation and returning
def give_hue(image):

    #separating into hue, saturation and value channels
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    #Display hue only
    if image_show:
        cv2.imshow('hue', h)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return h

# K-means color quantization
def give_kmeans_cq(image, K):
    #conertinng to euclidean
    z = image.reshape((-1, 3))
    z = np.float32(z)

    #setting kmeans criteria and findinf k means
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 5,1.0)
    ret, label, center = cv2.kmeans(z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    #applying color quantization using kmeans data
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape(image.shape)

    #display kmeans cq result
    if image_show:
        cv2.imshow('res2', res2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return res2

def green_mask(image, h_value):

    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    sensitivity = 30;
    lower_green = np.array([h_value - sensitivity, 35, 10])
    upper_green = np.array([h_value + sensitivity, 255, 255])
    mask = cv2.inRange(image, lower_green, upper_green)
    mask = cv2.bitwise_not(mask)
    image = cv2.bitwise_and(image, image, mask=mask)
    image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    # display kmeans cq result
    if image_show:
        cv2.imshow('after green mask', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return image


def get_features(original):
    image = original.copy()

    if image_show:
        cv2.imshow('original', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Bilaterall filter for slight blurring and noise cancelling
    image = cv2.bilateralFilter(image, 9, 75, 75)


    # Get Kmeans color quantized image
    image = give_kmeans_cq(image, 12)


    # Apply a green mask
    image = green_mask(image)

    if image_show:
        cv2.imshow('bf', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return image


