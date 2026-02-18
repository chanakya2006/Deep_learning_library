CXX := g++
CXXFLAGS := -std=c++20 -O2 -Wall -Wextra -Iinclude -MMD -MP -g

SRC_DIR := src
BUILD_DIR := build

SRCS := $(shell find $(SRC_DIR) -name "*.cpp") main.cpp
OBJS := $(SRCS:%.cpp=$(BUILD_DIR)/%.o)
DEPS := $(OBJS:.o=.d)

TARGET := train

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $^ -o $@

$(BUILD_DIR)/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

-include $(DEPS)

clean:
	rm -rf $(BUILD_DIR) $(TARGET)

.PHONY: all clean
