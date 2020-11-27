void sync_gpu_buffer(void* dst, const void* src, size_t count)
{
	cudaMemcpy ( dst, src, count, cudaMemcpyHostToDevice);
}

void sync_cpu_buffer(void* dst, const void* src, size_t count)
{
	cudaMemcpy ( dst, src, count, cudaMemcpyDeviceToHost);
}