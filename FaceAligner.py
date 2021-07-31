import cv2
import numpy as np
import os
import argparse
import glob
import sys
import dlib

#create argparse
parser = argparse.ArgumentParser(description='Make all face pictures level')
parser.add_argument('--directory', required=True, help='Directory of project (contains folder with images and classification files)')
parser.add_argument('--img_folder', required=True, help='Name of the folder containing images')
parser.add_argument('--img_name', help='Name of test image')
parser.add_argument('--video', action='store_true', help='Do you want to convert finished images into a video?')
args = parser.parse_args()

numAlteredImages = 0

def convertToVideo(finished_path):
    global numAlteredImages
    print("Generating Video")
    sys.stdout.write('['+' '*numAlteredImages+']  0%')
    sys.stdout.flush()
    img_array = []
    for entry in os.scandir(finished_path):
        img = cv2.imread(entry.path)
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)
    
    out = cv2.VideoWriter('PictureADay.mp4', 0x7634706d, 10.0, size)
    count = 0
    for i in range(len(img_array)):
        out.write(img_array[i])
        sys.stdout.write('\b'*((numAlteredImages+5)-count) + '=')
        if(count < (numAlteredImages - 1)):
            sys.stdout.write('>')
        sys.stdout.write(' '*((numAlteredImages-2)-count) + '] ' + str(int(((count+1)/numAlteredImages)*10)) + '0%')
        sys.stdout.flush()
        count+=1
        
    sys.stdout.write('\b\b\b\bDone!\n')
    out.release()

def processFolder(directory_path, img_folder, make_video):
    print(directory_path)
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

    #Check if image folder is populated
    if not os.listdir(imgs_path):
        print("IMAGE FOLDER IS EMPTY")
        return
    #iterate through all images in folder and fix
    for entry in os.scandir(imgs_path):
        img_name = entry.path.split(os.sep)[-1]
        #straightenImage(img_name, entry.path, directory_path, failed_path, face_cascade, eye_cascade, finished_path)
        dlibstraighten(img_name, entry.path, directory_path, failed_path, face_cascade, eye_cascade, finished_path)
    
    if(make_video):
        convertToVideo(finished_path)


def getMainFace(faces, center):
    face_index = 0;
    faces_centers= []
    if(len(faces)>1):
        for i in faces:
            face_center = (int((i[0][0]+i[1][0])/2),int((i[0][1]+i[1][1])/2))
            diff_center = (abs(center[0]-face_center[0]), abs(center[1]-face_center[1]))
            faces_centers.append(diff_center)
        #Find smallest distance from center of face to center of image
        smallest = faces_centers[0]
        for i in range(len(faces_centers)):
            if(faces_centers[i]<smallest):
                smallest = faces_centers[i]
                face_index = i
    return face_index

def dlibstraighten(img_name, img_path, directory, failed_path, face_cascade, eye_cascade, finished_path):
    global numAlteredImages
    #loading the image
    img = cv2.imread(img_path)
    os.chdir(directory)
    

    newFilename = img_name[:-4] + 'Aligned' + '.jpg'

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    detector = dlib.get_frontal_face_detector()
    detections = detector(img, 1)

    sp = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    faces = dlib.full_object_detections()
    for det in detections:
        faces.append(sp(img, det))
    if(len(faces)>1):
        print("More than 1 face:", img_name)
        return
    # Bounding box and eyes
    bb = [i.rect for i in faces]
    bb = [((i.left(), i.top()),
       (i.right(), i.bottom())) for i in bb]                            # Convert out of dlib format

    imgd = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if len(faces) == 0:
#         #detection failed
        print("Face not detected:", img_name)
        cv2.imwrite(os.path.join(failed_path, img_name),imgd)
        return
    # Width and height of the image
    h, w = imgd.shape[:2]
    # Calculating a center point of the image
    center = (w // 2, h // 2)
    
    # Find main face, get index for that face and following features
    mf_index = getMainFace(bb, center)

    right_eyes = [[face.part(i) for i in range(36, 42)] for face in faces]
    right_eyes = [[(i.x, i.y) for i in eye] for eye in right_eyes]          # Convert out of dlib format

    left_eyes = [[face.part(i) for i in range(42, 48)] for face in faces]
    left_eyes = [[(i.x, i.y) for i in eye] for eye in left_eyes]            # Convert out of dlib format

    if(len(bb)>2):
        #get the top left and bottom right point of eye to find center point
        print("More than 1 face!")
        left_eye_bb_cord = (max(left_eyes[mf_index], key=lambda x: x[0])[0], max(left_eyes[mf_index], key=lambda x: x[1])[1])
        right_eye_bb_cord = (max(right_eyes[mf_index], key=lambda x: x[0])[0], max(right_eyes[mf_index], key=lambda x: x[1])[1])
    else:
        left_eye_bb_cord = (max(left_eyes, key=lambda x: x[0])[0], max(left_eyes, key=lambda x: x[1])[1])
        right_eye_bb_cord = (max(right_eyes, key=lambda x: x[0])[0], max(right_eyes, key=lambda x: x[1])[1])

    #Find center point between two points
    left_eye_center = (int((left_eye_bb_cord[0][0]+left_eye_bb_cord[1][0])/2),int((left_eye_bb_cord[0][1]+left_eye_bb_cord[1][1])/2))
    left_eye_x = left_eye_center[0] 
    left_eye_y = left_eye_center[1]

    right_eye_center = (int((right_eye_bb_cord[0][0]+right_eye_bb_cord[1][0])/2),int((right_eye_bb_cord[0][1]+right_eye_bb_cord[1][1])/2))
    right_eye_x = right_eye_center[0]
    right_eye_y = right_eye_center[1]
    
    if left_eye_y > right_eye_y:
        A = (right_eye_x, left_eye_y)
        # Integer -1 indicates that the image will rotate in the clockwise direction
        direction = -1 
    else:
        A = (left_eye_x, right_eye_y)
        # Integer 1 indicates that image will rotate in the counter clockwise direction 
        direction = 1 

    delta_x = right_eye_x - left_eye_x
    delta_y = right_eye_y - left_eye_y
    angle=np.arctan(delta_y/delta_x)
    angle = (angle * 180) / np.pi

    eyes_midpoint = ((left_eye_x + right_eye_x)//2, (left_eye_y + right_eye_y)//2)

    #DISPLAY
    # for i in bb:
    #     cv2.rectangle(imgd, i[0], i[1], (255, 0, 0), 5)     # Bounding box

    # for eye in right_eyes:
    #     cv2.rectangle(imgd, (max(eye, key=lambda x: x[0])[0], max(eye, key=lambda x: x[1])[1]),
    #                     (min(eye, key=lambda x: x[0])[0], min(eye, key=lambda x: x[1])[1]),
    #                     (0, 0, 255), 5)
    #     for point in eye:
    #         cv2.circle(imgd, (point[0], point[1]), 2, (0, 255, 0), -1)

    # for eye in left_eyes:
    #     cv2.rectangle(imgd, (max(eye, key=lambda x: x[0])[0], max(eye, key=lambda x: x[1])[1]),
    #                     (min(eye, key=lambda x: x[0])[0], min(eye, key=lambda x: x[1])[1]),
    #                     (0, 255, 0), 5)
    #     for point in eye:
    #         cv2.circle(imgd, (point[0], point[1]), 2, (0, 0, 255), -1)


    # Defining a matrix M and calling cv2.getRotationMatrix2D method
    M = cv2.getRotationMatrix2D(center, (angle), 1.0)
    # Applying the rotation to our image 
    rotated = cv2.warpAffine(imgd, M, (w, h))

    # calculate distance between the eyes in the first image

    ratio = 1
    # Defining aspect ratio of a resized image
    dim = (int(w * ratio), int(h * ratio))
    # We have obtained a new image that we call resized3
    resized = cv2.resize(rotated, dim)


    move_x = (center[0] - eyes_midpoint[0])
    move_y = (center[1] - eyes_midpoint[1])

    translation_matrix = np.array([
        [1, 0, move_x],
        [0, 1, move_y]
    ], dtype=np.float32)

    translated_image = cv2.warpAffine(src=resized, M=translation_matrix, dsize=(w,h))


    cv2.imwrite(os.path.join(finished_path, newFilename),translated_image)
    print(newFilename +":","Done")
    numAlteredImages+=1

# def straightenImage(img_name, img_path, directory, failed_path, face_cascade, eye_cascade, finished_path):
#     global numAlteredImages
#     #loading the image
#     img = cv2.imread(img_path)
#     os.chdir(directory)

#     newFilename = img_name[:-4] + 'Aligned' + '.jpg'


#     # Converting the image into grayscale
#     gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     # Creating variable faces
#     faces= face_cascade.detectMultiScale (gray, 1.1, 4)
#     if len(faces) == 0:
#         #detection failed
#         print("Face not detected:", img_name)
#         cv2.imwrite(os.path.join(failed_path, img_name),img)
#         return

#     # Defining the rectangle around the face
#     x, y, w, h = faces[0]

#     # Creating two regions of interest
#     roi_gray=gray[y:(y+h), x:(x+w)]
#     roi_color=img[y:(y+h), x:(x+w)]

#     # Creating variable eyes
#     eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 4)
#     if(len(eyes)<2):
#         print("Failed eye detection:", img_name)
#         cv2.imwrite(os.path.join(failed_path, img_name),img)
#         return

#     index=0
#     # Creating for loop in order to divide one eye from another
#     for (ex , ey,  ew,  eh) in eyes:
#         if index == 0:
#             eye_1 = (ex, ey, ew, eh)
#         elif index == 1:
#             eye_2 = (ex, ey, ew, eh)

#         index = index + 1

#     if eye_1[0] < eye_2[0]:
#         left_eye = eye_1
#         right_eye = eye_2
#     else:
#         left_eye = eye_2
#         right_eye = eye_1

#     # Calculating coordinates of a central points of the rectangles
#     left_eye_center = (int(left_eye[0] + (left_eye[2] / 2)), int(left_eye[1] + (left_eye[3] / 2)))
#     left_eye_x = left_eye_center[0] 
#     left_eye_y = left_eye_center[1]
 
#     right_eye_center = (int(right_eye[0] + (right_eye[2]/2)), int(right_eye[1] + (right_eye[3]/2)))
#     right_eye_x = right_eye_center[0]
#     right_eye_y = right_eye_center[1]
    
#     if left_eye_y > right_eye_y:
#         A = (right_eye_x, left_eye_y)
#         # Integer -1 indicates that the image will rotate in the clockwise direction
#         direction = -1 
#     else:
#         A = (left_eye_x, right_eye_y)
#         # Integer 1 indicates that image will rotate in the counter clockwise direction 
#         direction = 1 

#     delta_x = right_eye_x - left_eye_x
#     delta_y = right_eye_y - left_eye_y
#     angle=np.arctan(delta_y/delta_x)
#     angle = (angle * 180) / np.pi

#     eyes_midpoint = ((left_eye_x + right_eye_x)//2, (left_eye_y + right_eye_y)//2)

#     # Width and height of the image
#     h, w = img.shape[:2]
#     # Calculating a center point of the image
#     center = (w // 2, h // 2)

#     # Defining a matrix M and calling cv2.getRotationMatrix2D method
#     M = cv2.getRotationMatrix2D(center, (angle), 1.0)
#     # Applying the rotation to our image 
#     rotated = cv2.warpAffine(img, M, (w, h))

#     # calculate distance between the eyes in the first image

#     ratio = 1
#     # Defining aspect ratio of a resized image
#     dim = (int(w * ratio), int(h * ratio))
#     # We have obtained a new image that we call resized3
#     resized = cv2.resize(rotated, dim)

#     #centered = center_image(resized, face_cascade, eye_cascade)
#     xEye = x + eyes_midpoint[0]
#     yEye = y + eyes_midpoint[1]
    
#     move_x = (center[0] - xEye)
#     move_y = (center[1] - yEye)

#     translation_matrix = np.array([
#         [1, 0, move_x],
#         [0, 1, move_y]
#     ], dtype=np.float32)

#     translated_image = cv2.warpAffine(src=resized, M=translation_matrix, dsize=(w,h))


#     cv2.imwrite(os.path.join(finished_path, newFilename),translated_image)
#     print(newFilename +":","Done")
#     numAlteredImages+=1




if __name__ == '__main__':
    processFolder(args.directory, args.img_folder, args.video)
    


