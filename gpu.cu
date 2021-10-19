#include "gpu.cuh"

#include <iostream>

#define MAX_MASS 150
#define MAX_COMPRESS 0.5
#define MIN_MASS 1
#define FLOW_STRENGHT 1

__device__ void SetPixel(uint8_t* surfaceData, int index, uint8_t r, uint8_t g, uint8_t b) {
	surfaceData[index * 4] = b;
	surfaceData[index * 4 + 1] = g;
	surfaceData[index * 4 + 2] = r;
}

__device__ int clamp(int a, int min, int max) {
	return a > max ? max : a < min ? min : a;
}

__device__ uint8_t GetType(uint8_t* simData, int index) {
	return simData[index * 2];
}

__device__ uint8_t GetMass(uint8_t* simData, int index) {
	return simData[index * 2 + 1];
}

__device__ void SetType(uint8_t* simData, int index, uint8_t type) {
	simData[index * 2] = type;
}

__device__ void SetMass(uint8_t* simData, int index, uint8_t mass) {
	simData[index * 2 + 1] = mass;
}

__device__ void AddMass(uint8_t* simData, int index, uint8_t mass) {
	simData[index * 2 + 1] += mass;
}

__global__ void DrawGPU(uint8_t* surfaceData, uint8_t* simData) {
	int currentIndex = threadIdx.x + (blockIdx.x * blockDim.x);

	switch (simData[currentIndex * 2]) {
	case 1: {
		SetPixel(surfaceData, currentIndex, 0, 0, GetMass(simData, currentIndex));
		break;
	}
	case 2: {
		SetPixel(surfaceData, currentIndex, 168, 96, 50);
		break;
	}
	default: {
		SetPixel(surfaceData, currentIndex, 0, 0, 0);
		break;
	}
	}
}

__global__ void ComputeGPU(uint8_t* curSimData, uint8_t* nextSimData, int* size) {
	int w = size[0];
	int h = size[1];

	int currentIndex = threadIdx.x + (blockIdx.x * blockDim.x);
	int belowIndex = currentIndex + w;
	int aboveIndex = currentIndex - w;
	int rightIndex = currentIndex + 1;
	int leftIndex = currentIndex - 1;

	int rightBelowIndex = currentIndex + 1 + w;
	int rightAboveIndex = currentIndex + 1 - w;
	int leftBelowIndex = currentIndex - 1 + w;
	int leftAboveIndex = currentIndex - 1 - w;

	if (GetType(curSimData, currentIndex) > 1) {
		SetType(nextSimData, currentIndex, GetType(curSimData, currentIndex));
		SetMass(nextSimData, currentIndex, GetMass(curSimData, currentIndex));
		return;
	}

	uint8_t currentMass = GetMass(curSimData, currentIndex);

	SetMass(nextSimData, currentIndex, currentMass);

	if (GetType(curSimData, aboveIndex) < 2) {
		uint8_t aboveMass = GetMass(curSimData, aboveIndex);
		uint8_t currentCapacity = clamp((MAX_MASS + (int)aboveMass * MAX_COMPRESS) - (int)currentMass, 0, 255);
		int flow = clamp(currentCapacity, -currentMass, currentMass + aboveMass <= 255 ? aboveMass : 255 - currentMass);
		flow *= FLOW_STRENGHT;
		AddMass(nextSimData, currentIndex, flow);
	}

	int belowFlow = 0;

	if (GetType(curSimData, belowIndex) < 2) {
		uint8_t belowMass = GetMass(curSimData, belowIndex);
		uint8_t belowCapacity = clamp((MAX_MASS + (int)currentMass * MAX_COMPRESS) - (int)belowMass, 0, 255);
		belowFlow = clamp(belowCapacity, -belowMass, belowMass + currentMass <= 255 ? currentMass : 255 - belowMass);
		belowFlow *= FLOW_STRENGHT;
		AddMass(nextSimData, currentIndex, -belowFlow);
	}

	uint8_t rightMass = GetMass(curSimData, rightIndex);
	uint8_t leftMass = GetMass(curSimData, leftIndex);

	uint8_t newCurrentMass = GetMass(nextSimData, currentIndex);
	uint8_t newRightMass = rightMass;
	uint8_t newLeftMass = leftMass;
	//right Update

	if (GetType(curSimData, rightAboveIndex) < 2) {
		uint8_t rightAboveMass = GetMass(curSimData, rightAboveIndex);
		uint8_t rightCurrentCapacity = clamp((MAX_MASS + (int)rightAboveMass * MAX_COMPRESS) - (int)rightMass, 0, 255);
		int flow = clamp(rightCurrentCapacity, -rightMass, rightMass + rightAboveMass <= 255 ? rightAboveMass : 255 - rightMass);
		flow *= FLOW_STRENGHT;
		newRightMass += flow;
	}

	if (GetType(curSimData, rightBelowIndex) < 2) {
		uint8_t rightBelowMass = GetMass(curSimData, rightBelowIndex);
		uint8_t rightBelowCapacity = clamp((MAX_MASS + (int)rightMass * MAX_COMPRESS) - (int)rightBelowMass, 0, 255);
		int rightBelowFlow = clamp(rightBelowCapacity, -rightBelowMass, rightBelowMass + rightMass <= 255 ? rightMass : 255 - rightBelowMass);
		rightBelowFlow *= FLOW_STRENGHT;
		newRightMass -= rightBelowFlow;
	}

	//end
	//left Update

	if (GetType(curSimData, leftAboveIndex) < 2) {
		uint8_t leftAboveMass = GetMass(curSimData, leftAboveIndex);
		uint8_t leftCurrentCapacity = clamp((MAX_MASS + (int)leftAboveMass * MAX_COMPRESS) - (int)leftMass, 0, 255);
		int flow = clamp(leftCurrentCapacity, -leftMass, leftMass + leftAboveMass <= 255 ? leftAboveMass : 255 - leftMass);
		flow *= FLOW_STRENGHT;
		newLeftMass += flow;
	}

	if (GetType(curSimData, leftBelowIndex) < 2) {
		uint8_t leftBelowMass = GetMass(curSimData, leftBelowIndex);
		uint8_t leftBelowCapacity = clamp((MAX_MASS + (int)leftMass * MAX_COMPRESS) - (int)leftBelowMass, 0, 255);
		int leftBelowFlow = clamp(leftBelowCapacity, -leftBelowMass, leftBelowMass + leftMass <= 255 ? leftMass : 255 - leftBelowMass);
		leftBelowFlow *= FLOW_STRENGHT;
		newLeftMass -= leftBelowFlow;
	}

	//end

	if (belowFlow < 2 || newCurrentMass > MAX_MASS) {
		if (GetType(curSimData, rightIndex) < 2) {
			int rflow = (newRightMass - newCurrentMass) / 2 * FLOW_STRENGHT;
			AddMass(nextSimData, currentIndex, rflow);
		}
		if (GetType(curSimData, leftIndex) < 2) {
			int lflow = (newLeftMass - newCurrentMass) / 2 * FLOW_STRENGHT;
			AddMass(nextSimData, currentIndex, lflow);
		}
	}

	if (GetMass(nextSimData, currentIndex) > MIN_MASS) {
		SetType(nextSimData, currentIndex, 1);
	}
	else {
		SetType(nextSimData, currentIndex, 0);
	}
}

__host__ void* GPU_Alloc(size_t size)
{
	void* ptr = nullptr;
	cudaMallocManaged(&ptr, size);
	return ptr;
}

__host__ void GPU_Free(void* ptr) 
{
	cudaFree(ptr);
}

__host__ void ComputeSimulation(void* current, void* next, int* size, unsigned int w, unsigned int h) {
	dim3 blockN{ (w * h) / 1024 };
	dim3 threadN{ 1024 };

	ComputeGPU<<<blockN, threadN>>>((uint8_t*)current, (uint8_t*)next, size);

	cudaDeviceSynchronize();
}

__host__ void Draw(void* surfaceData, void* simData, unsigned int w, unsigned int h) {
	dim3 blockN{ (w * h) / 1024 };
	dim3 threadN{ 1024 };

	DrawGPU<<<blockN, threadN>>>((uint8_t*)surfaceData, (uint8_t*)simData);

	cudaDeviceSynchronize();
}
