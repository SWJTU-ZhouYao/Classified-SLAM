cmake_minimum_required( VERSION 3.2 )
project( yolov5-seg-example )

set( CMAKE_BUILD_TYPE "Release" )
set( CMAKE_CXX_FLAGS "-std=c++14 -march=native -O3" )

# CUDA and cuDNN
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
find_package(CUDA REQUIRED)
message(STATUS " libraries: ${CUDA_LIBRARIES}")
message(STATUS " include path: ${CUDA_INCLUDE_DIRS}")
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)
list(APPEND CUDA_NVCC_FLAGS -std=c++14;-O3)
list(APPEND CUDA_NVCC_FLAGS -Xcompiler;-fPIC)
LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_86,code=sm_86) # match your computer
enable_language(CUDA)

# TensorRT
list(APPEND CMAKE_PREFIX_PATH "~/c++_packages/TensorRT-8.4.3.1")
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
        HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
        PATH_SUFFIXES include)
MESSAGE(STATUS "Found TensorRT headers at ${TENSORRT_INCLUDE_DIR}")
find_library(TENSORRT_LIBRARY_INFER nvinfer
        HINTS ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
        PATH_SUFFIXES lib lib64 lib/x64)
find_library(TENSORRT_LIBRARY_INFER_PLUGIN nvinfer_plugin
        HINTS  ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
        PATH_SUFFIXES lib lib64 lib/x64)
set(TENSORRT_LIBRARY ${TENSORRT_LIBRARY_INFER} ${TENSORRT_LIBRARY_INFER_PLUGIN})
MESSAGE(STATUS "Find TensorRT libs at ${TENSORRT_LIBRARY}")
find_package_handle_standard_args(
        TENSORRT DEFAULT_MSG TENSORRT_INCLUDE_DIR TENSORRT_LIBRARY)
if(NOT TENSORRT_FOUND)
    message(ERROR "Cannot find TensorRT library.")
endif()

find_package( OpenCV REQUIRED )
message(STATUS "opencv version" ${OpenCV_VERSION})

include_directories(
        ${TENSORRT_INCLUDE_DIR}
        ${OpenCV_INCLUDE_DIRS}
        "/usr/include/eigen3/"
)
add_executable(main main.cpp)
target_link_libraries(main ${OpenCV_LIBS} ${TENSORRT_LIBRARY} cudart)