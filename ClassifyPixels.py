from PixelClassifier import *
from Preprocessing import *
import cv2

# Load image to be classified
target_im = cv2.imread("./Train/img.png")
give_Segmentation(target_im)
# # Load training images
# unclass_im = cv2.imread("./Train/t1.png")
# class_im = cv2.imread("./Train/t2.png")
#
# # Initialize PixelClassifier instance and set training images
# pc = PixelClassifier()
# pc.set_training_images(im_class=class_im, im_orig=unclass_im)
#
# # Tune classifier
# pc.tune(n_estimators_list=[10], filtername_list=["bilateral"], filter_d_list=range(5,30,5), filter_sigmacolor_list=range(20,100,20), filter_sigmaspace_list=range(20,100,20), cv=5)
#
#
#
# # Classify
# final_im = pc.predict(target_im)
#
# cv2.imshow('original', final_im)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# # Save result
# cv2.imwrite('Train/imgC1.png', final_im)