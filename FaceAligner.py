import cv2
import numpy as np
import os
import argparse
import glob

#create argparse
parser = argparse.ArgumentParser(description='Make all face pictures level')
parser.add_argument('--directory', required=True, help='Directory of project (contains folder with images and classification files)')
parser.add_argument('--img_folder', required=True, help='Name of the folder containing images')
parser.add_argument('--img_name', help='Name of test image')
parser.add_argument('--make_video', required=True, help='Do you want to convert finished images into a video? T or F')
args = parser.parse_args()

#creating face_cascade and eye_cascade objects
#face_cascade = cv2.CascadeClassifier('C:\\Users\\hosfo\\Documents\\GitHub\\PictureADayAligner\\haarcascade_frontalface_default.xml')
#eye_cascade = cv2.CascadeClassifier('C:\\Users\\hosfo\\Documents\\GitHub\\PictureADayAligner\\haarcascade_eye.xml')

#failed_path = 'C:\\Users\\hosfo\\Documents\\GitHub\\PictureADayAligner\\FailedDetections'
#img_name = 'test2Brandon.jpg'
#img_path = 'C:\\Users\\hosfo\\Documents\\GitHub\\PictureADayAligner' + '\\' + img_name
#directory = r'C:\Users\hosfo\Documents\GitHub\PictureADayAligner'

def convertToVideo(finished_path):
    img_array = []
    for entry in os.scandir(finished_path):
        img = cv2.imread(entry.path)
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)
    
    out = cv2.VideoWriter('PictureADay.mp4', 0x7634706d, 10.0, size)
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

def processFolder(directory_path, img_folder, make_video):
    print(directory_path)
    #os.chdir(directory_path)
    #creating face_cascade and eye_cascade objects
    face_cascade = cv2.CascadeClassifier(directory_path + r'\haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(directory_path + r'\haarcascade_eye.xml')

    imgs_path = os.path.join(directory_path, img_folder)
    #Create failed detections and fixed images folder if it doesn't already exist
    if not os.path.exists('FailedDetections'):
        os.makedirs('FailedDetections')
    if not os.path.exists('FixedImages'):
        os.makedirs('FixedImages')
    failed_path = directory_path + r'\FailedDetections'
    finished_path = directory_path + r'\FixedImages'

    #iterate through all images in folder and fix
    for entry in os.scandir(imgs_path):
        img_name = entry.path.split(os.sep)[-1]
        straightenImage(img_name, entry.path, directory_path, failed_path, face_cascade, eye_cascade, finished_path)
    
    if(make_video=='T' or make_video == 'True'):
        convertToVideo(finished_path)

def straightenImage(img_name, img_path, directory, failed_path, face_cascade, eye_cascade, finished_path):
    #loading the image
    img = cv2.imread(img_path)
    os.chdir(directory)

    newFilename = img_name[:-4] + 'Aligned' + '.jpg'


    # Converting the image into grayscale
    gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Creating variable faces
    faces= face_cascade.detectMultiScale (gray, 1.1, 4)
    if len(faces) == 0:
        #detection failed
        print("Face not detected:", img_name)
        cv2.imwrite(os.path.join(failed_path, img_name),img)
        return

    # Defining and drawing* the rectangle around the face
    x, y, w, h = faces[0]

    # Creating two regions of interest
    roi_gray=gray[y:(y+h), x:(x+w)]
    roi_color=img[y:(y+h), x:(x+w)]

    # Creating variable eyes
    eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 4)
    if(len(eyes)<2):
        print("Failed eye detection:", img_name)
        cv2.imwrite(os.path.join(failed_path, img_name),img)
        return

    index=0
    # Creating for loop in order to divide one eye from another
    for (ex , ey,  ew,  eh) in eyes:
        if index == 0:
            eye_1 = (ex, ey, ew, eh)
        elif index == 1:
            eye_2 = (ex, ey, ew, eh)

        index = index + 1

    if eye_1[0] < eye_2[0]:
        left_eye = eye_1
        right_eye = eye_2
    else:
        left_eye = eye_2
        right_eye = eye_1

    # Calculating coordinates of a central points of the rectangles
    left_eye_center = (int(left_eye[0] + (left_eye[2] / 2)), int(left_eye[1] + (left_eye[3] / 2)))
    left_eye_x = left_eye_center[0] 
    left_eye_y = left_eye_center[1]
 
    right_eye_center = (int(right_eye[0] + (right_eye[2]/2)), int(right_eye[1] + (right_eye[3]/2)))
    right_eye_x = right_eye_center[0]
    right_eye_y = right_eye_center[1]
    
    #cv2.circle(roi_color, left_eye_center, 5, (255, 0, 0) , -1)
    #cv2.circle(roi_color, right_eye_center, 5, (255, 0, 0) , -1)
    #cv2.line(roi_color,right_eye_center, left_eye_center,(0,200,200),3)

    if left_eye_y > right_eye_y:
        A = (right_eye_x, left_eye_y)
        # Integer -1 indicates that the image will rotate in the clockwise direction
        direction = -1 
    else:
        A = (left_eye_x, right_eye_y)
        # Integer 1 indicates that image will rotate in the counter clockwise direction 
        direction = 1 

    #cv2.circle(roi_color, A, 5, (255, 0, 0) , -1)
    
    #cv2.line(roi_color,right_eye_center, left_eye_center,(0,200,200),3)
    #cv2.line(roi_color,left_eye_center, A,(0,200,200),3)
    #cv2.line(roi_color,right_eye_center, A,(0,200,200),3)

    delta_x = right_eye_x - left_eye_x
    delta_y = right_eye_y - left_eye_y
    angle=np.arctan(delta_y/delta_x)
    angle = (angle * 180) / np.pi

    eyes_midpoint = ((left_eye_x + right_eye_x)//2, (left_eye_y + right_eye_y)//2)

    
    

    # Width and height of the image
    h, w = img.shape[:2]
    # Calculating a center point of the image
    center = (w // 2, h // 2)

    # Defining a matrix M and calling
    # cv2.getRotationMatrix2D method
    M = cv2.getRotationMatrix2D(center, (angle), 1.0)
    # Applying the rotation to our image 
    rotated = cv2.warpAffine(img, M, (w, h))

    # calculate distance between the eyes in the first image
    dist_1 = np.sqrt((delta_x * delta_x) + (delta_y * delta_y))

    ratio = 1


    # Defining aspect ratio of a resized image
    dim = (int(w * ratio), int(h * ratio))
    # We have obtained a new image that we call resized3
    resized = cv2.resize(rotated, dim)

    #centered = center_image(resized, face_cascade, eye_cascade)
    xEye = x + eyes_midpoint[0]
    yEye = y + eyes_midpoint[1]
    
    move_x = (center[0] - xEye)
    move_y = (center[1] - yEye)

    translation_matrix = np.array([
        [1, 0, move_x],
        [0, 1, move_y]
    ], dtype=np.float32)

    translated_image = cv2.warpAffine(src=resized, M=translation_matrix, dsize=(w,h))


    cv2.imwrite(os.path.join(finished_path, newFilename),translated_image)
    print("Done")




if __name__ == '__main__':
    processFolder(args.directory, args.img_folder, args.make_video)
    


