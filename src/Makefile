
debug		:= 0
include		../Makefile.inc

all: clean VectorAdd Flops

VectorAdd: VectorAdd.cpp
	$(INTELCXX) -o $(PROJ_BASE)/bin/VectorAdd VectorAdd.cpp $(INCLUDES) $(LIBS) $(CFLAGS) $(LDFLAGS) 

Flops: Flops.cpp
	$(INTELCXX) -o $(PROJ_BASE)/bin/Flops Flops.cpp $(INCLUDES) $(LIBS) $(CFLAGS) $(LDFLAGS) 

clean:
	rm -f $(PROJ_BASE)/bin/VectorAdd
	rm -f $(PROJ_BASE)/bin/Flops
