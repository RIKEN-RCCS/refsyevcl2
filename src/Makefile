
# c++
CXX = mpiFCCpx
FLAGS = -Kfast,parallel,openmp -Nlst -O3 -MD -DTIMING_REF\
		-SCALAPACK -SSL2BLAMP \
		-I. 

# FORTRUN
FC = mpifrtpx
FFLAGS = -Kfast,parallel,openmp \
		 -SCALAPACK -SSL2BLAMP \
		 -I../EigenExa-2.12/include \
		 -L../EigenExa-2.12/lib -lEigenExa

OBJS = \
	eigen_exa_interfaces.o \
	error_free.o \
	ozmm.o \
	ir.o \
	testmat.o \
	eval.o

LIBNAME = librefsyev.a

%.o: %.cpp %.hpp
	$(CXX) $(FLAGS) -c $< -o $@

$(LIBNAME): $(OBJS)
	ar rv $(LIBNAME) $(OBJS)

eigen_exa_interfaces.o: eigen_exa_interfaces.F90
	$(FC) $(FFLAGS) -c eigen_exa_interfaces.F90 -o eigen_exa_interfaces.o

.PHONY: clean

.PHONY: FORCE

clean:
	rm -f *.o
	rm -f $(LIBNAME)
	rm *.d *.lst

all: $(LIBNAME)
