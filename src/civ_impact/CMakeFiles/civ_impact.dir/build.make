# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.21

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /data/stronglogic/kneural

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /data/stronglogic/kneural

# Include any dependencies generated for this target.
include src/civ_impact/CMakeFiles/civ_impact.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include src/civ_impact/CMakeFiles/civ_impact.dir/compiler_depend.make

# Include the progress variables for this target.
include src/civ_impact/CMakeFiles/civ_impact.dir/progress.make

# Include the compile flags for this target's objects.
include src/civ_impact/CMakeFiles/civ_impact.dir/flags.make

src/civ_impact/CMakeFiles/civ_impact.dir/main.cpp.o: src/civ_impact/CMakeFiles/civ_impact.dir/flags.make
src/civ_impact/CMakeFiles/civ_impact.dir/main.cpp.o: src/civ_impact/main.cpp
src/civ_impact/CMakeFiles/civ_impact.dir/main.cpp.o: src/civ_impact/CMakeFiles/civ_impact.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/data/stronglogic/kneural/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/civ_impact/CMakeFiles/civ_impact.dir/main.cpp.o"
	cd /data/stronglogic/kneural/src/civ_impact && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT src/civ_impact/CMakeFiles/civ_impact.dir/main.cpp.o -MF CMakeFiles/civ_impact.dir/main.cpp.o.d -o CMakeFiles/civ_impact.dir/main.cpp.o -c /data/stronglogic/kneural/src/civ_impact/main.cpp

src/civ_impact/CMakeFiles/civ_impact.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/civ_impact.dir/main.cpp.i"
	cd /data/stronglogic/kneural/src/civ_impact && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /data/stronglogic/kneural/src/civ_impact/main.cpp > CMakeFiles/civ_impact.dir/main.cpp.i

src/civ_impact/CMakeFiles/civ_impact.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/civ_impact.dir/main.cpp.s"
	cd /data/stronglogic/kneural/src/civ_impact && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /data/stronglogic/kneural/src/civ_impact/main.cpp -o CMakeFiles/civ_impact.dir/main.cpp.s

# Object files for target civ_impact
civ_impact_OBJECTS = \
"CMakeFiles/civ_impact.dir/main.cpp.o"

# External object files for target civ_impact
civ_impact_EXTERNAL_OBJECTS =

src/civ_impact/civ_impact: src/civ_impact/CMakeFiles/civ_impact.dir/main.cpp.o
src/civ_impact/civ_impact: src/civ_impact/CMakeFiles/civ_impact.dir/build.make
src/civ_impact/civ_impact: third_party/opennn/opennn/libopennn.a
src/civ_impact/civ_impact: src/civ_impact/CMakeFiles/civ_impact.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/data/stronglogic/kneural/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable civ_impact"
	cd /data/stronglogic/kneural/src/civ_impact && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/civ_impact.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/civ_impact/CMakeFiles/civ_impact.dir/build: src/civ_impact/civ_impact
.PHONY : src/civ_impact/CMakeFiles/civ_impact.dir/build

src/civ_impact/CMakeFiles/civ_impact.dir/clean:
	cd /data/stronglogic/kneural/src/civ_impact && $(CMAKE_COMMAND) -P CMakeFiles/civ_impact.dir/cmake_clean.cmake
.PHONY : src/civ_impact/CMakeFiles/civ_impact.dir/clean

src/civ_impact/CMakeFiles/civ_impact.dir/depend:
	cd /data/stronglogic/kneural && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /data/stronglogic/kneural /data/stronglogic/kneural/src/civ_impact /data/stronglogic/kneural /data/stronglogic/kneural/src/civ_impact /data/stronglogic/kneural/src/civ_impact/CMakeFiles/civ_impact.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/civ_impact/CMakeFiles/civ_impact.dir/depend

