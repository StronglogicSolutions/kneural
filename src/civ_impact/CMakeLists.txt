cmake_minimum_required(VERSION 2.8.12)

project(civ_impact)

	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
	set(OPENNNLIB "${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/opennn/opennn/libopennn.a")

	message(${CMAKE_CURRENT_SOURCE_DIR})
add_executable(civ_impact main.cpp)

target_link_libraries(civ_impact PUBLIC ${OPENNNLIB})

target_include_directories(${PROJECT_NAME} PRIVATE
  "../../third_party"
)