# LibCV

Small library containing data structures and algorithms for most basic computer vision tasks, 
developed as learning project and coding practice, to be used for my faculty projects.

## Contents
At this early stage, libcv contains following:
* Array structures - with reference counted data, and slicing operators.
	* template vector (with heap data), and vectorx (with statically defined size).
	* template matrix (m by n) designed for large matrices, that could hold images.
	* image array structure abstraction with type defined in runtime, which essentially is an 3D array.
* template bounded priority queue structure
* basic k-d tree structure implementation with k-nearest neighbour search.
* 2D contour, polygon and region structures
* basic image manipulation algorithms.
* basic linear algebra operations.
* lmdif integration from cminpack.
* image i/o
* basic gui support for showing images

## Dependencies
Libcv depends on following libraries:
* Qt4 (4.8.6)
* libpng (libpng16)
* libjpg (jpegsr8)
* lapacke
* blas

## Documentation
Proper documentation is not available, and it will not be if there is no interest
from other people to use this library. I'll try to document code in doxygen style,
so if anyone is up to it, he can build doc files himself.

## Compilation
LibCV is mainly developed on Ubuntu with gcc 4.9.2. Ubuntu 15.04 has all needed libraries (and
corresponding versions) on it's repositories, which can be installed using apt-get.

It was not tested on other *nix systems, but as long as your able to use gcc
with c++11 support, you could easily download, build and link needed libraries.
Also building blas and lapack should not be of any trouble on any *nix system.

### Custom CMake build flags
* **WITH_GUI** - if FALSE, ignores gui module, and doesn't link Qt
* **WITH_JPEG** - turn on jpeg i/o support. If FALSE, does not link libjpeg.
* **DOUBLE_REAL** - if TRUE, real type in LibCV is double, if FALSE it is of single precision.
* **COMPILE_EXAMPLES** - compile example programs in ./examples/ directory. 

## Compilation on Windows
LibCV has been tested on Windows 8 with Visual Studio 2013. Instructions here will
guide compilation using these tools, but for any older VC should not be much different.

**NOTE:** In **./thirdParty/ directory are built libs using Visual Studio 2013 (VC12) x64**. If you need
to compile it using some other toolchain, here are instructions I've followed to compile
dependencies with vc12:

### Qt
Download Qt 4.8.6 source here:
http://download.qt.io/archive/qt/4.8/4.8.6/

Compilation steps:

1. Open VS Command line
2. configure -release -opensource -no-libpng -no-libjpeg
3. nmake
4. nmake install

**NOTE:** The **-no-libpng** and **-no-libjpeg** flags are entered since specified versions of 
png and jpeg libraries are used, Qt internals would not match, and image i/o
would not work - png would compile ok, but runtime error would be thrown on 
image loading, for jpeg I'm not sure, haven't test it.

### Png 
libpng 1.6 can be downloaded at sourceforge:
[http://sourceforge.net/projects/libpng/files/](http://sourceforge.net/projects/libpng/files/)

You'd need [zlib](http://www.zlib.net/) also. Both are easily compiled using CMake, 
and with **vc12** should not be any trouble.

### Jpeg
I've skipped compiling jpeg on windows. There's some instructions on the web how to build
it using nmake, but I've had no luck. If anyone can help with that, please mail me, I'd be
grateful.

### OpenBLAS
Easiest way to get blas and lapacke is through OpenBlas. Here are some nice instructions
on how to install it:
[https://github.com/arrayfire/arrayfire/wiki/CBLAS-for-Windows](https://github.com/arrayfire/arrayfire/wiki/CBLAS-for-Windows)


As guide suggests, you'd need cygwin with x86_64-w64-mingw32-gcc and x86_64-w64-mingw32-gfortran,
and with cygwin terminal, compilation is easy - just make, and make install:

```
make BINARY=64 CC=/usr/bin/x86_64-w64-mingw32-gcc-4.8.3.exe FC=/usr/bin/x86_64-w64-mingw32-gfortran.exe
make PREFIX=./package install
```

Step 8., 9. and 10. can be skipped, since resulting lib file (libopenblas.dll.a) can be 
successfully linked with VC.

# Release Versions
Version 0.1 is tagged because it was the version which [camera calibration](https://github.com/ljubobratovicrelja/camera_calibration)
is developed with. But I can't guarantee the stability of library inside - it is only preserved because of the camera calibration project.
 
# Contributions
If anybody is interested to make a contrubution, you're welcome, and thank you! 
Of course at this point, coding conventions or such rules are not firmly 
established - as long as you're willing to contribute, just make a pull request, 
and we'll work it out...

# LICENSE
Library is under MIT license. See LICENSE file for details.
