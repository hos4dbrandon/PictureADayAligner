import cv2
import numpy as np
import os
#from google.colab.patches import cv2_imshow

#creating face_cascade and eye_cascade objects
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")

img_path = r'C:\Users\hosfo\Documents\GitHub\PictureADayAligner\testBrandon.jpg'
directory = r'C:\Users\hosfo\Documents\GitHub\PictureADayAligner'

#loading the image
img = cv2.imread(img_path)
os.chdir(directory)

newFilename = 'savedImg.jpg'

cv2.imwrite(newFilename,img)


#cv2.imshow('testBrandon',img)
print("done")