
PROJ_BASE	:= $(HOME)/src/Benchmarks

# https://github.com/isazi/utils
ISA_UTILS_INCLUDE	:= $(HOME)/src/utils/include
ISA_UTILS_SRC		:= $(HOME)/src/utils/src/utils.cpp
# https://github.com/isazi/OpenCL
ISA_OPENCL_INCLUDE	:= $(HOME)/src/OpenCL/include

INCLUDES	:= -I"$(PROJ_BASE)/include" -I"$(ISA_UTILS_INCLUDE)"
LIBS		:= 
OPENCL 		:= -I"$(OPENCL_INCLUDE)" -I"$(ISA_OPENCL_INCLUDE)" -L"$(OPENCL_LIB)"

ifneq ($(debug), 1)
	CFLAGS		:= -std=c++0x -O3 -g0 -Wall -fopenmp
else
	CFLAGS		:= -std=c++0x -O0 -g3 -Wall
endif
CFLAGS_SSE		:= $(CFLAGS) -msse2 
CFLAGS_AVX		:= $(CFLAGS) -avx
LDFLAGS         := -lm
LDFLAGS_OPENCL	:= $(LDFLAGS) -lOpenCL

CC			:= g++
LINKER		:= g++

all: clean Copy CopySSE Read Write

Copy: Copy.cpp
	$(LINKER) -o $(PROJ_BASE)/bin/CopyBenchmark $(INCLUDES) $(LIBS) $(OPENCL) Copy.cpp $(CFLAGS) $(LDFLAGS_OPENCL)

CopySSE: CopySSE.cpp
	$(LINKER) -o $(PROJ_BASE)/bin/CopySSEBenchmark $(INCLUDES) $(LIBS) CopySSE.cpp $(CFLAGS_SSE) $(LDFLAGS)

Read: Read.cpp
	$(LINKER) -o $(PROJ_BASE)/bin/ReadBenchmark $(INCLUDES) $(LIBS) $(OPENCL) Read.cpp $(CFLAGS) $(LDFLAGS_OPENCL)

Write: Write.cpp
	$(LINKER) -o $(PROJ_BASE)/bin/WriteBenchmark $(INCLUDES) $(LIBS) $(OPENCL) Write.cpp $(CFLAGS) $(LDFLAGS_OPENCL)

clean:
	rm -f $(PROJ_BASE)/bin/CopyBenchmark
	rm -f $(PROJ_BASE)/bin/CopySSEBenchmark
	rm -f $(PROJ_BASE)/bin/ReadBenchmark
	rm -f $(PROJ_BASE)/bin/WriteBenchmark
