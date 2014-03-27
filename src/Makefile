
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

all: clean Copy CopySSE CopyAVX Read ReadSSE ReadAVX Write WriteSSE WriteAVX VectorAdd

Copy: Copy.cpp
	$(LINKER) -o $(PROJ_BASE)/bin/CopyBenchmark $(INCLUDES) $(LIBS) $(OPENCL) Copy.cpp $(CFLAGS) $(LDFLAGS_OPENCL)

CopySSE: CopySSE.cpp
	$(LINKER) -o $(PROJ_BASE)/bin/CopySSEBenchmark $(INCLUDES) $(LIBS) CopySSE.cpp $(CFLAGS_SSE) $(LDFLAGS)

CopyAVX: CopyAVX.cpp
	$(LINKER) -o $(PROJ_BASE)/bin/CopyAVXBenchmark $(INCLUDES) $(LIBS) CopyAVX.cpp $(CFLAGS_AVX) $(LDFLAGS)

Read: Read.cpp
	$(LINKER) -o $(PROJ_BASE)/bin/ReadBenchmark $(INCLUDES) $(LIBS) $(OPENCL) Read.cpp $(CFLAGS) $(LDFLAGS_OPENCL)

ReadSSE: ReadSSE.cpp
	$(LINKER) -o $(PROJ_BASE)/bin/ReadSSEBenchmark $(INCLUDES) $(LIBS) ReadSSE.cpp $(CFLAGS_SSE) $(LDFLAGS)

ReadAVX: ReadAVX.cpp
	$(LINKER) -o $(PROJ_BASE)/bin/ReadAVXBenchmark $(INCLUDES) $(LIBS) ReadAVX.cpp $(CFLAGS_AVX) $(LDFLAGS)

Write: Write.cpp
	$(LINKER) -o $(PROJ_BASE)/bin/WriteBenchmark $(INCLUDES) $(LIBS) $(OPENCL) Write.cpp $(CFLAGS) $(LDFLAGS_OPENCL)

WriteSSE: WriteSSE.cpp
	$(LINKER) -o $(PROJ_BASE)/bin/WriteSSEBenchmark $(INCLUDES) $(LIBS) WriteSSE.cpp $(CFLAGS_SSE) $(LDFLAGS)

WriteAVX: WriteAVX.cpp
	$(LINKER) -o $(PROJ_BASE)/bin/WriteAVXBenchmark $(INCLUDES) $(LIBS) WriteAVX.cpp $(CFLAGS_AVX) $(LDFLAGS)

VectorAdd: VectorAdd.cpp
	$(LINKER) -o $(PROJ_BASE)/bin/VectorAddBenchmark $(INCLUDES) $(LIBS) $(OPENCL) VectorAdd.cpp $(CFLAGS) $(LDFLAGS_OPENCL)

Flops: Flops.cpp ../include/Flops.hpp
	$(LINKER) -o $(PROJ_BASE)/bin/FlopsBenchmark $(INCLUDES) $(LIBS) $(OPENCL) Flops.cpp $(CFLAGS) $(LDFLAGS_OPENCL)

clean:
	rm -f $(PROJ_BASE)/bin/CopyBenchmark
	rm -f $(PROJ_BASE)/bin/CopySSEBenchmark
	rm -f $(PROJ_BASE)/bin/CopyAVXBenchmark
	rm -f $(PROJ_BASE)/bin/ReadBenchmark
	rm -f $(PROJ_BASE)/bin/ReadSSEBenchmark
	rm -f $(PROJ_BASE)/bin/ReadAVXBenchmark
	rm -f $(PROJ_BASE)/bin/WriteBenchmark
	rm -f $(PROJ_BASE)/bin/WriteSSEBenchmark
	rm -f $(PROJ_BASE)/bin/WriteAVXBenchmark
	rm -f $(PROJ_BASE)/bin/VectorAddBenchmark
	rm -f $(PROJ_BASE)/bin/FlopsBenchmark
