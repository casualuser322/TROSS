cmake_minimum_required(VERSION 3.16)
project(TROSS_Orchestrator VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON) 

find_package(OpenCV REQUIRED)
message(STATUS "Found OpenCV: ${OpenCV_VERSION} at ${OpenCV_INCLUDE_DIRS}")

find_package(Python3 COMPONENTS Development REQUIRED)
message(STATUS "Found Python3: ${Python3_INCLUDE_DIRS}")

add_library(orchestrator SHARED
    src/orchestrator.cpp
    src/bindings.cpp
)

target_include_directories(orchestrator
    PUBLIC 
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${OpenCV_INCLUDE_DIRS}  
        ${Python3_INCLUDE_DIRS} 
)

target_link_libraries(orchestrator
    ${OpenCV_LIBS}          
    ${Python3_LIBRARIES}    
)

install(TARGETS orchestrator
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)


install(DIRECTORY include/ DESTINATION include)
