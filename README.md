# PictureADayAligner
Aligns face pictures using OpenCV 


Need to install OpenCV, cv2 and dlib in order for program to work.

Put images to be aligned in a folder

When running script, be sure to include the following required arguments:

--directory : path to directory where running project

--img_folder : Name of folder containing images (within --directory specified above)

--video :flag indicating if mp4 animation of final images is desired

example: 

python FaceAligner.py --directory C:\Users\hosfo\Documents\GitHub\PictureADayAligner --img_folder RawImages --video

LIMITATIONS: Currently can't deal with multiple faces. If encounters image with 2 faces, will ignore that image.
  TODO: Implement a GUI to select desired face if image with multiple faces encountered 
