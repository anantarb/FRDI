cmake_minimum_required(VERSION 3.6)

project(Face_Reconstruction)

set(LIBRARY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../Libs CACHE PATH "Path to lib folder")
set(Eigen3_DIR ${LIBRARY_DIR}/Eigen/share/eigen3/cmake CACHE PATH "Path to installed Eigen")
set(HDF5_DIR ${LIBRARY_DIR}/HDF5/cmake CACHE PATH "Path to installed HDF5")
set(glog_DIR ${LIBRARY_DIR}/Glog/lib/cmake/glog CACHE PATH "Path to installed glog")
set(Ceres_DIR ${LIBRARY_DIR}/Ceres/CMake CACHE PATH "Path to installed Ceres")
set(OpenCV_DIR ${LIBRARY_DIR}/OpenCV CACHE PATH "Path to installed OpenCV")

#(For Ananta cuz of MACOS)
#set(Ceres_DIR ${LIBRARY_DIR}/Ceres/lib/cmake/Ceres CACHE PATH "Path to installed Ceres")
#set(OpenCV_DIR ${LIBRARY_DIR}/opencv/lib/cmake/opencv4 CACHE PATH "Path to installed OpenCV")

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

find_package(HDF5 REQUIRED C CXX)
include_directories(${HDF5_INCLUDE_DIRS})
if(WIN32)
    get_target_property(HDF5_DLL_PATH_DEBUG hdf5-shared IMPORTED_LOCATION_DEBUG)
    get_target_property(HDF5_DLL_PATH_RELEASE hdf5-shared IMPORTED_LOCATION_RELEASE)
endif(WIN32)

find_package(glog REQUIRED)
include_directories(${glog_INCLUDE_DIR})
if(WIN32)
    get_target_property(GLOG_DLL_PATH_DEBUG glog::glog IMPORTED_LOCATION_DEBUG)
    get_target_property(GLOG_DLL_PATH_RELEASE glog::glog IMPORTED_LOCATION_RELEASE)
endif(WIN32)

find_package(Ceres REQUIRED)
include_directories(${Ceres_INCLUDE_DIR})

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIR})

if(WIN32)
    get_target_property(OPENCV_CORE_DLL_PATH_DEBUG opencv_core IMPORTED_LOCATION_DEBUG)
    get_target_property(OPENCV_IMGPROC_DLL_PATH_DEBUG opencv_imgproc IMPORTED_LOCATION_DEBUG)
    get_target_property(OPENCV_IMGCODECS_DLL_PATH_DEBUG opencv_imgcodecs IMPORTED_LOCATION_DEBUG)
    get_target_property(OPENCV_VIDEOIO_DLL_PATH_DEBUG opencv_videoio IMPORTED_LOCATION_DEBUG)
    get_target_property(OPENCV_HIGHGUI_DLL_PATH_DEBUG opencv_highgui IMPORTED_LOCATION_DEBUG)    
    get_target_property(OPENCV_CORE_DLL_PATH_RELEASE opencv_core IMPORTED_LOCATION_RELEASE)
    get_target_property(OPENCV_IMGPROC_DLL_PATH_RELEASE opencv_imgproc IMPORTED_LOCATION_RELEASE)
    get_target_property(OPENCV_IMGCODECS_DLL_PATH_RELEASE opencv_imgcodecs IMPORTED_LOCATION_RELEASE)
    get_target_property(OPENCV_VIDEOIO_DLL_PATH_RELEASE opencv_videoio IMPORTED_LOCATION_RELEASE)
    get_target_property(OPENCV_HIGHGUI_DLL_PATH_RELEASE opencv_highgui IMPORTED_LOCATION_RELEASE)
endif(WIN32)

# Set C++ flags
set(CMAKE_CXX_STANDARD 17)

add_definitions("-D_DISABLE_EXTENDED_ALIGNED_STORAGE")

set(HEADER_FILE
    include/Eigen.h
    include/model.h
    include/preprocessor.h
	include/landmarks.h
    include/obj_loader
)

set(SOURCE_FILE
    src/utils.cpp
    src/model.cpp
    src/preprocessor.cpp
	src/landmarks.cpp
)

add_executable(Face_Reconstruction main.cpp ${HEADER_FILE} ${SOURCE_FILE})
target_link_libraries(Face_Reconstruction Eigen3::Eigen ${HDF5_LIBRARIES} ${HDF5_CXX_LIBRARIES} ${OpenCV_LIBS} ceres)

# On Windows copy dll to executable folder
if(WIN32)
    # Visual Studio properties
    set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT Face_Reconstruction)
    set_property(TARGET Face_Reconstruction PROPERTY VS_DEBUGGER_WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/)

    # On Windows copy dll to executable folder
    add_custom_command(TARGET Face_Reconstruction POST_BUILD        
        COMMAND ${CMAKE_COMMAND} -E copy_if_different          
            "${HDF5_DLL_PATH_DEBUG}"                           
            ${CMAKE_BINARY_DIR})                     
    
    add_custom_command(TARGET Face_Reconstruction POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${HDF5_DLL_PATH_RELEASE}"
            ${CMAKE_BINARY_DIR})
    
    add_custom_command(TARGET Face_Reconstruction POST_BUILD        
        COMMAND ${CMAKE_COMMAND} -E copy_if_different          
            "${GLOG_DLL_PATH_DEBUG}"                           
            ${CMAKE_BINARY_DIR})                     
        
    add_custom_command(TARGET Face_Reconstruction POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${GLOG_DLL_PATH_RELEASE}"
            ${CMAKE_BINARY_DIR})
    
    add_custom_command(TARGET Face_Reconstruction POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${OPENCV_CORE_DLL_PATH_DEBUG}"
            ${CMAKE_BINARY_DIR})
    
    add_custom_command(TARGET Face_Reconstruction POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${OPENCV_IMGPROC_DLL_PATH_DEBUG}"
            ${CMAKE_BINARY_DIR})
    
    add_custom_command(TARGET Face_Reconstruction POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${OPENCV_IMGCODECS_DLL_PATH_DEBUG}"
            ${CMAKE_BINARY_DIR})
    
    add_custom_command(TARGET Face_Reconstruction POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${OPENCV_VIDEOIO_DLL_PATH_DEBUG}"
            ${CMAKE_BINARY_DIR})
    
    add_custom_command(TARGET Face_Reconstruction POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${OPENCV_HIGHGUI_DLL_PATH_DEBUG}"
            ${CMAKE_BINARY_DIR})
    
    add_custom_command(TARGET Face_Reconstruction POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${OPENCV_CORE_DLL_PATH_RELEASE}"
            ${CMAKE_BINARY_DIR})
    
    add_custom_command(TARGET Face_Reconstruction POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${OPENCV_IMGPROC_DLL_PATH_RELEASE}"
            ${CMAKE_BINARY_DIR})
    
    add_custom_command(TARGET Face_Reconstruction POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${OPENCV_IMGCODECS_DLL_PATH_RELEASE}"
            ${CMAKE_BINARY_DIR})
    
    add_custom_command(TARGET Face_Reconstruction POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${OPENCV_VIDEOIO_DLL_PATH_RELEASE}"
            ${CMAKE_BINARY_DIR})
    
    add_custom_command(TARGET Face_Reconstruction POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${OPENCV_HIGHGUI_DLL_PATH_RELEASE}"
            ${CMAKE_BINARY_DIR})
endif(WIN32)