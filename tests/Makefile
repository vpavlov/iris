NVCC=nvcc
FLAGS=-O0 -g
all:
	$(NVCC) $(FLAGS) -I../src -DIRIS_CUDA -o memory_gpu.o -c ../src/memory_gpu.cu
	$(NVCC) $(FLAGS) -I../src -DIRIS_CUDA -o test_buffer_manager.o  -c test_buffer_manager.cu
	$(NVCC) $(FLAGS) -o test_buffer_manager test_buffer_manager.o memory_gpu.o
