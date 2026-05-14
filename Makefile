CXX      ?= clang++
CXXFLAGS ?= -std=c++20 -O2 -Wall -Wextra -pthread
LDFLAGS  ?= -pthread

SRC := src/baseline.cpp
HDR := src/mpmc_queue.hpp src/spsc_queue.hpp

all: baseline

baseline: $(SRC) $(HDR)
	$(CXX) $(CXXFLAGS) $(SRC) -o $@ $(LDFLAGS)

baseline-tsan: $(SRC) $(HDR)
	$(CXX) $(CXXFLAGS) -fsanitize=thread -g $(SRC) -o $@ $(LDFLAGS) -fsanitize=thread

clean:
	rm -f baseline baseline-tsan

.PHONY: all clean
