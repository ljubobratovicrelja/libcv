# LibCV

Small library containing data structures and algorithms for basic computer vision tasks, 
developed as learning project, and coding practice, to be used for my faculty projects.

## Dependencies
Libcv depends on following libraries:
* Qt4 (4.8.6) - for gui (if compiled with CV_IGNORE_GUI Qt is not required)
* libpng (libpng16)
* libjpg (jpegsr8)
* lapacke
* blas

### Qt4 note
Qt 4.8.6 is used for this project. In linux, this version comes by default in most distributions'
package mangers. For windows, this version needs to be downloaded, compiled with corresponding 
compiler and tools, and linked in cmake for libcv compilation.

# LICENSE
Library is under MIT license. See LICENSE file for details.
