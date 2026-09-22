#########################################################################
# GStamps: C++ routines for the Local & the Global Postage Stamp Problem
# Authors:
#   L. Colisson, J-G. Dumas, A. Galan, B. Grenet, A. Maignan, D. S. Roche
#########################################################################


OPTFLAGS = -O3 -ffast-math

#######
# g++ options
OPTFLAGS += -fopenmp
OPTFLAGS += -fprefetch-loop-arrays -funroll-all-loops
OPTFLAGS += -UDEBUG -DNDEBUG
#OPTFLAGS +=  -D__GSTAMPS_EXTENDED_PRECISION
#######


CXXFLAGS += ${OPTFLAGS} -I`pwd`/include/ `pkg-config givaro --cflags`
LOADLIBES+= `pkg-config givaro --libs`

#######

PRG  = range basis
PRG += dynprg supplement complement
PRG += brute
PRG += search
PRG += incremental

BEN  = fibo alba geom bala
BEN += krange reach srange
BEN += depthrange decompositions

#######

EXE = ${PRG} ${BEN}
BIN = ${EXE:%=bin/%}

all: ${BIN}

prg: ${PRG:%=bin/%}

bench: ${BEN:%=bin/%}

VPATH = src:benchmarks

bin/%: %.cpp
	$(LINK.cpp) $^ $(LOADLIBES) $(LDLIBS) -o $@

bin/incremental_parallel: src/incremental_parallel.cpp src/incremental.cpp
	$(LINK.cpp) $< $(LOADLIBES) $(LDLIBS) -o $@

# S18: bin/incremental itself builds profile-guided (fastest measured exact
# solver, ~-20%). Training is k=8 serial by default (same hot code as 9/4;
# override with PGO_TRAIN_ARGS="9 4 serial" for the last ~3%).
# Set INCREMENTAL_PGO=0 for a plain -O3 build while iterating on kernels.
PGODIR = ./.pgo-data
PGO_TRAIN_ARGS ?= 8 4 serial
ifeq ($(INCREMENTAL_PGO),0)
bin/incremental: src/incremental.cpp
	$(LINK.cpp) $^ $(LOADLIBES) $(LDLIBS) -o $@
else
bin/incremental: src/incremental.cpp
	mkdir -p ${PGODIR}
	$(LINK.cpp) -fprofile-generate=${PGODIR} $^ $(LOADLIBES) $(LDLIBS) -o $@
	./$@ ${PGO_TRAIN_ARGS}
	$(LINK.cpp) -fprofile-use=${PGODIR} $^ $(LOADLIBES) $(LDLIBS) -o $@
endif

TBB_CXXFLAGS ?= $(shell pkg-config --cflags tbb 2>/dev/null)
TBB_LIBS ?= $(shell pkg-config --libs tbb 2>/dev/null)

tbb: bin/incremental_tbb

bin/incremental_tbb: src/incremental_tbb.cpp src/incremental.cpp
	$(LINK.cpp) $(TBB_CXXFLAGS) $< $(LOADLIBES) $(TBB_LIBS) $(LDLIBS) -o $@

clean:
	- \rm ${BIN}
	- \rm -rf ${PGODIR}

range: FDTC.sh ${BIN}
	./$< 5 3 1 3 1 6
	./$< 5 5 0 5 1 6
	./$< 30 3 1 3 20 6
	./$< 50 8 1 3 45 5
	./$< 5 50 0 5 45 4

basis: FDTB.sh ${BIN}
	./$< 15

brute: FDTA.sh ${BIN}
	./$< 10

decompositions: FDTF.sh ${BEN}
	./$< 9 9

check: brute basis decompositions range
