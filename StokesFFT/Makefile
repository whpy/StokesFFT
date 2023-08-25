OS := $(shell uname)

SFFT:
	nvcc -lcufft -o SFFT StokesFFT.cu 

struct:
	nvcc -lcufft -o SFFT StokesFFT-structured.cu
wclean:
	del *.lib
	del *.exp
	del *.csv
	del *.exe

clean:
	rm SFFT
	rm *.csv
