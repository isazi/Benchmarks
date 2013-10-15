
PROJ_BASE	:= $(HOME)/src/Benchmarks

# https://github.com/isazi/utils
ISA_UTILS_INCLUDE	:= $(HOME)/src/utils/include
ISA_UTILS_SRC		:= $(HOME)/src/utils/src/utils.cpp
# https://github.com/isazi/OpenCL
ISA_OPENCL_INCLUDE	:= $(HOME)/src/OpenCL/include

INCLUDES	:= -I"$(PROJ_BASE)/include" -I"$(OPENCL_INCLUDE)" -I"$(ISA_UTILS_INCLUDE)" -I"$(ISA_OPENCL_INCLUDE)"
LIBS		:= -L"$(OPENCL_LIB)"

ifneq ($(debug), 1)
	CFLAGS		:= -std=c++0x -O3 -g0 -Wall
else
	CFLAGS		:= -std=c++0x -O0 -g3 -Wall
endif

LDFLAGS         := -lm -lOpenCL

CC			:= g++
LINKER		:= g++

all: clean Copy Read

Copy: Copy.cpp
	$(LINKER) -o $(PROJ_BASE)/bin/CopyBenchmark $(INCLUDES) $(LIBS) Copy.cpp $(CFLAGS) $(LDFLAGS)

Read: Read.cpp
	$(LINKER) -o $(PROJ_BASE)/bin/ReadBenchmark $(INCLUDES) $(LIBS) Read.cpp $(CFLAGS) $(LDFLAGS)

clean:
	rm -f $(PROJ_BASE)/bin/CopyBenchmark
	rm -f $(PROJ_BASE)/bin/ReadBenchmark
