# Makefile for a C++ program using Clang with C++20 and Matplotlibcpp on Windows

# Compiler and flags
CXX = clang
CXXFLAGS = -std=c++20 -O3 -lmatplot

# Matplotlibcpp header file path
MATPLOTLIBCPP_INCLUDE = -I"Matplot++ 1.2.0\include" -L"Matplot++ 1.2.0\lib"

# Source files
SRCS = tsp.cpp

# Executable name
TARGET = tsp

# Rule to build the executable
$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) $(MATPLOTLIBCPP_INCLUDE) $(SRCS) -o $(TARGET)

# Rule to clean the project
clean:
	rm -f $(TARGET)

# Rule to run the program
run: $(TARGET)
	./$(TARGET)
