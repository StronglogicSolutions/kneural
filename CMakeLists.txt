cmake_minimum_required(VERSION 2.8.12)

project(kneural)

set(CMAKE_CXX_STANDARD 17)
set(CPACK_GENERATOR "TGZ")
set(CMAKE_SUPPRESS_REGENERATION true)

add_subdirectory(third_party/opennn)
add_subdirectory(src)
include_directories(third_party/opennn)
include_directories(src)

include(CPack)