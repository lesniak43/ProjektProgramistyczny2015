AAM/POSIT face detector
by Damian Leśniak, 6th Jun 2015


GENERAL INFORMATION:

This software was strongly inspired by:
"Mastering OpenCV with Practical Computer Vision Projects", Chapter 7
https://github.com/MasteringOpenCV/code/tree/master/Chapter7_HeadPoseEstimation by Daniel Lélis Baggio

Author of the original AAM method: Tim Cootes (Professor of Computer Vision)

Creator ot the 3D face model used in this software: Pedro A. D. Martins (PhD)
http://aifi.isr.uc.pt/Downloads/OpenGL/glAnthropometric3DModel.cpp

Comprehensive description of the AAM algorithm:
P. Martins, "Active Appearance Models for Facial Expression Recognition and Monocular Head Pose Estimation"


HOW TO INSTALL:

cd path_to_project_directory
cmake .
make


BINARIES:

get_data:
use the mouse to drag points across the face snapshot
'c' - start/pause webcam
's' - save changes (warning! all unsaved changes are lost if you unpause the webcam or use keys '>', '<')
'i' / 'u' - rotate shape (warning! this action, if performed repeatedly, results in distiortions of points' coordinates)
'j' / 'k' / 'l' / 'h' - move shape
'>' / '<' - go to next/previous shapshot
'+' - make a copy of current snapshot and append it to the list of images
'e' - equalize the snapshot (not recommended)
'z' - show/hide edges
'x' - show/hide vertices

train:
creates AAM and stores it in the data directory

test_model:
you may check out the parameters learned by AAM and see if POSIT is working correctly

final:
sit back and enjoy AAM/POSIT struggling to recognize you


HOW TO CREATE A WORKING AAM/POSIT:

Option 1:
    - grow a beard
    - run the 'train' executable

Option 2:
    - go to the 'images' directory
    - remove all but one images (leave '1.jpg')
    - go to the 'data' directory
    - edit the 'shapes.txt' file
        - change the first line to '1 116'
        - don't modify the second line
        - delete all the following lines
    - run the 'get_data' executable to modify the first image and create at least 20 new images
    - run the 'train' executable
    - run the 'test_model' executable to check if everything went according to the plan

Option 3 (advanced, not tested, might require additional changes in the source code):
    - create your own 3D shape, sort the vertices
    - go to the 'data' directory and modify the following files (the first line should always be: 'number_of_rows number_of_cols'
        - model3D.txt - store point coordinates in each row
        - reference_shape.txt - project the 3D model into 2D space, store the coordinates in one row: 'x1 y1 x2 y2 ... xn yn'
        - shapes.txt - replace with a copy of 'reference_shape.txt'
        - triangles.txt - triangulate your 3D model, store each triangle (described by its vertices' numbers) in one row
    - edit the 'main.cpp' source file, delete (or replace) the code using 'face_cascade' to estimate position of the face
    - go to the 'images' directory
    - remove all but one images (leave '1.jpg')
    - run the 'get_data' executable to modify the first image and create new images
    - run the 'train' executable
    - run the 'test_model' executable to check if everything went according to the plan


TUNABLE PARAMETERS:
    - in 'main.cpp': MAX_ALLOWED_ENERGY - used to check if AAM approximation is good enough
