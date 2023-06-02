exe_file = main
cuda_file = mmulUnified

all:
	nvcc -o $(exe_file) $(cuda_file).cu

clean:
	rm -f $(exe_file)
