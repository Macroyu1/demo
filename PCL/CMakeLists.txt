cmake_minimum_required(VERSION 3.16)
project(PCL_Test LANGUAGES CXX)
 
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
 
find_package(PCL)
include_directories(${PCL_INCLUDE_DIRS})
#头文件和源文件都放在src目录下
file(GLOB HEADER_FILES ${CMAKE_CURRENT_SOURCE_DIR}/src/*.h*)
file(GLOB SOURCES_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*.c*)
 
add_executable(${PROJECT_NAME}
    ${HEADER_FILES}
    ${SOURCES_FILES} 
) 
 
target_link_libraries(${PROJECT_NAME} PRIVATE 
    ${PCL_LIBRARIES}
)