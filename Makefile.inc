
PROJ_BASE	:= $(HOME)/src/Benchmarks

# https://github.com/isazi/utils
ISA_UTILS_INCLUDE	:= $(HOME)/src/utils/include
# https://github.com/isazi/OpenCL
ISA_OPENCL_INCLUDE	:= $(HOME)/src/OpenCL/include

INCLUDES := -I"$(PROJ_BASE)/include" -I"$(OPENCL_INCLUDE)" -I"$(ISA_UTILS_INCLUDE)" -I"$(ISA_OPENCL_INCLUDE)"
CPU_INCLUDES := -I"$(PROJ_BASE)/include" -I"$(ISA_UTILS_INCLUDE)"
PHI_INCLUDES := -I"$(PROJ_BASE)/include" -I"$(ISA_UTILS_INCLUDE)"
LIBS := -L"$(OPENCL_LIB)"

ifneq ($(debug), 1)
	CFLAGS := -std=c++11 -O3 -g0 -Wall
	CPU_CFLAGS := -std=c++11 -O3 -g0 -Wall -mavx -fopenmp
	PHI_CFLAGS := -std=c++11 -O3 -g0 -Wall -mmic -openmp -vec-report3
else
	CFLAGS := -std=c++11 -O0 -g3 -Wall
	CPU_CFLAGS := -std=c++11 -O0 -g3 -Wall -mavx -fopenmp
	PHI_CFLAGS := -std=c++11 -O0 -g3 -Wall -mmic -openmp -vec-report3
endif

LDFLAGS := -lm -lOpenCL
CPU_LDFLAGS := -lm
PHI_LDFLAGS := -lm

CC := g++
INTELCXX := icc

