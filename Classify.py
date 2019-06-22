from Preprocessing import *
import cv2

original = cv2.imread("./Train/1.png")
segments = give_segmentation(original)
# for each in segments:
