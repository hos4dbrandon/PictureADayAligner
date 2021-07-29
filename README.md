# PictureADayAligner
Aligns face pictures using OpenCV 


Need to install OpenCV in order for program to work.

Put images to be aligned in a folder

When running script, be sure to include the following required arguments:
--directory : path to directory where running project
--img_folder : Name of folder containing images (within --directory specified above)
--make_video : Boolean indicating if mp4 animation of final images is desired (T or F)

example: 
python FaceAligner.py --directory C:\Users\hosfo\Documents\GitHub\PictureADayAligner --img_folder RawImages --make_video T
