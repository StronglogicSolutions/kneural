cmake_minimum_required(VERSION 2.8.12)

project(civ_impact)

if(UNIX)
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
	set(PROJECT_LINK_LIBS ${CMAKE_SOURCE_DIR}/Release/opennn/libopennn.a)
endif()

add_executable(civ_impact main.cpp)

target_link_libraries(civ_impact PUBLIC opennn)

target_include_directories(${PROJECT_NAME} PRIVATE
  "../../third_party"
)