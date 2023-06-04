cuda_file = mmulUnified

all:
	nvcc -o $(cuda_file) $(cuda_file).cu

clean:
	rm -f $(cuda_file)
