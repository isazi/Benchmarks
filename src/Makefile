
debug		:= 0
include		../Makefile.inc

all: clean VectorAdd

VectorAdd: VectorAdd.cpp
	$(INTELCXX) -o $(PROJ_BASE)/bin/VectorAdd VectorAdd.cpp $(INCLUDES) $(LIBS) $(CFLAGS) $(LDFLAGS) 

clean:
	rm -f $(PROJ_BASE)/bin/VectorAdd
