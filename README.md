HOW TO INSTALL:

cd path_to_project_directory
cmake .
make


BINARIES:

get_data:

use the mouse to drag points across the face snapshot

c - start/pause webcam
s - save changes (warning! all unsaved changes are lost if you unpause the webcam or use keys '>', '<')
i/u - rotate shape (warning! this action, if performed repeatedly, results in distiortions of points' coordinates)
j/k/l/h - move shape
>/< - go to next/previous shapshot
+ - make a copy of current snapshot and append it to the list of images
e - equalize the snapshot (not recommended)
z - show/hide edges
x - show/hide vertices


train:

creates AAM and stores it in the data directory


test_model:

you may check out the parameters learned by AAM and see if POSIT is working correctly


final:

sit back and enjoy AAM/POSIT struggling to recognize you
