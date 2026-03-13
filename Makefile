CXX := g++

CXXFLAGS := -O3 -march=armv8-a+simd -std=c++17 -Wall

OPENCV_FLAGS := $(shell pkg-config --cflags --libs opencv4)

LIBS := -lpapi -pthread

TARGET := sobel_video_papi_headless
SRC := sobel_video_papi_headless.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(OPENCV_FLAGS) $(LIBS)

clean:
	rm -f $(TARGET)