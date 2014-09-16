
debug		:= 0
include		../Makefile.inc

all: clean VectorAdd Flops Flops4x2 Flops2x4 Bandwidth

VectorAdd: VectorAdd.cpp
	$(INTELCXX) -o $(PROJ_BASE)/bin/VectorAdd VectorAdd.cpp $(INCLUDES) $(LIBS) $(CFLAGS) $(LDFLAGS) 

Flops: Flops.cpp
	$(INTELCXX) -o $(PROJ_BASE)/bin/Flops Flops.cpp $(INCLUDES) $(LIBS) $(CFLAGS) $(LDFLAGS) 

Flops4x2: Flops4x2.cpp
	$(INTELCXX) -o $(PROJ_BASE)/bin/Flops4x2 Flops4x2.cpp $(INCLUDES) $(LIBS) $(CFLAGS) $(LDFLAGS) 

Flops2x4: Flops2x4.cpp
	$(INTELCXX) -o $(PROJ_BASE)/bin/Flops2x4 Flops2x4.cpp $(INCLUDES) $(LIBS) $(CFLAGS) $(LDFLAGS) 

Bandwidth: Bandwidth.cpp
	$(INTELCXX) -o $(PROJ_BASE)/bin/Bandwidth Bandwidth.cpp $(INCLUDES) $(LIBS) $(CFLAGS) $(LDFLAGS) 

clean:
	rm -f $(PROJ_BASE)/bin/VectorAdd
	rm -f $(PROJ_BASE)/bin/Flops
	rm -f $(PROJ_BASE)/bin/Flops4x2
	rm -f $(PROJ_BASE)/bin/Flops2x4
	rm -f $(PROJ_BASE)/bin/Bandwidth

