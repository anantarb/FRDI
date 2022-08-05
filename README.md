# 3DSMC-Project

# 1 Repository Organization
 * `include/`: All the header files go here.
 * `src/`: All the source files go here.
 * `main.cpp`: Main script to run the project.'
 * `../Data/`: All required files goes here (i.e. model2019\_bfm.h5, landmarks\_indices.txt, ...).
 * `../Libs/`: All installed libraries goes here.

# 2 Requirements
 * Eigen3
 * HDF5 C Components
 * glog
 * Ceres
 * OpenCV
 * face-alignment
 
#### 2.1 Windows Installation
 During CMake Installation for each Library, change `CMAKE_INSTALL_PREFIX` to the following:
 * `Eigen3`: `../Libs/Eigen`
 * `HDF5`: `../Libs/HDF5`
 * `glog`: `../Libs/Glog`
 * `Ceres`: `../Libs/Ceres`
 * `OpenCV`: `../Libs/OpenCV`
###### Notes:
 * Uncheck `BUILD_TESTING` for all of the above requirements.
 * Make sure to build `HDF5` & `glog` for both `debug` & `release`.
 * Ofcourse as long as the above libraries exist on the system somewhere else, they will be located automatically.


# 3 Running the project
 * Clone the repository.
 * Place 3DMM basel face model 2019 `model2019_bfm.h5` inside `../Data/`.
 * Adjust library paths inside `CMakeLists.txt`, if needed.
 * Compile the project using CMake.
 * For the first time, install conda environment for face-alignment by running

    ```
    conda create --name 3dfa python=3.7
    conda activate 3dfa
    pip install -r fa_requirements.txt
    ```
 * Run the project.
 * Face model `face.off` will be extracted inside `build` folder.
###### Notes:
 * Example code in main.cpp (Check "test_*" methods).
 * Copy `Libs\OpenCV\x64\vc17\bin\opencv_videoio_ffmpeg460_64.dll` To Project build dir