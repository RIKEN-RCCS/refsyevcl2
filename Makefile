
CXX = mpiFCCpx
FLAGS = -Kfast,parallel,openmp -Nlst -O3 -MD -SCALAPACK -SSL2BLAMP -DTIMING_REF
INCLUDES = -I../EigenExa-2.12/include \
           -I../src 
LIBS = -L../EigenExa-2.12/lib -lEigenExa \
	   -L../src -lrefsyev 

PROGRAM	= test_refsyevcl

all: $(PROGRAM)

$(PROGRAM): $(PROGRAM).cpp
	$(CXX) $< -o $@  $(FLAGS) $(LIBS) $(INCLUDES)

# sh:
# 	rm -f *.sh.*
# 	rm -r output.*

clean:
	rm -f *.o
	rm -f $(PROGRAM)
	rm -f *.d *.lst
