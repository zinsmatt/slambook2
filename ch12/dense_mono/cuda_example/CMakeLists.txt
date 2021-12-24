
cmake_minimum_required(VERSION 2.8)
project(cuda_example LANGUAGES CXX CUDA)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11 -march=native -O3")

############### dependencies ######################
# Eigen
include_directories("/usr/include/eigen3")

# # OpenCV
# find_package(OpenCV 3.1 REQUIRED)
# include_directories(${OpenCV_INCLUDE_DIRS})
# # Sophus
# find_package(Sophus REQUIRED)
# include_directories(${Sophus_INCLUDE_DIRS})


add_executable(${PROJECT_NAME} main.cu)
set_target_properties(${PROJECT_NAME}
                      PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
 
#target_link_libraries(${PROJECT_NAME} ${THIRD_PARTY_LIBS})
