#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__host__ void* GPU_Alloc(size_t size);
__host__ void GPU_Free(void* ptr);
__host__ void ComputeSimulation(void* current, void* next, int* size, unsigned int w, unsigned int h);
__host__ void Draw(void* surfaceData, void* simData, unsigned int w, unsigned int h);
