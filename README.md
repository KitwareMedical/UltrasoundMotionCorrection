# Respiratory motion correction for ultrasound images

Contains various attempts to correct for respiratory motion in bubble
ultrasound images. The currently best workin approach is the SliceToSmoothSliceBSpline code.

Contains code copied from https://bitbucket.org/suppechasper/elasticwarp .

Compilation is setup through [CMake](https://cmake.org/) and 
requires [ITK](www.itk.org) and [TCLAP](http://tclap.sourceforge.net/).

