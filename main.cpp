#define SDL_MAIN_HANDLED

#include <iostream>
#include <windows.h>
#include <stdexcept>
#include <SDL.h>
#include <SDL_ttf.h>

#include "gpu.cuh"

struct Container {
	SDL_Window* window;
	SDL_Renderer* renderer;
	SDL_Surface* surface;
	void* surfaceData;
	void* currentSimData;
	void* nextSimData;
	int* size;
};

Container Init(const char* name, int wW, int wH) {
	if (wW % 2 != 0 || wH % 2 != 0)
		throw std::runtime_error("Window size must be even.");

	Container container{};

	if (SDL_InitSubSystem(SDL_INIT_VIDEO))
		throw std::runtime_error("Unable to initialize SDL VIDEO sub system.");

	container.window = SDL_CreateWindow(name, SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
		wW, wH, SDL_WINDOW_SHOWN);

	if (container.window == nullptr)
		throw std::runtime_error("Unable to create window.");

	container.renderer = SDL_CreateRenderer(container.window, -1, SDL_RENDERER_ACCELERATED);

	if (container.renderer == nullptr)
		throw std::runtime_error("Unable to create renderer.");
	
	container.surface = SDL_CreateRGBSurface(0, wW, wH, 32, 0, 0, 0, 0);

	if (container.surface == nullptr)
		throw std::runtime_error("Unable to create surface.");

	SDL_SetSurfaceRLE(container.surface, 1);

	container.surfaceData = GPU_Alloc((size_t)container.surface->h * container.surface->pitch);
	container.currentSimData = GPU_Alloc((size_t)container.surface->w * container.surface->h * sizeof(uint16_t));
	container.nextSimData = GPU_Alloc((size_t)container.surface->w * container.surface->h * sizeof(uint16_t));
	container.size = (int*)GPU_Alloc(sizeof(int) * 2);

	printf("Surface data addr : %p\nCurrent simulation data : %p\nNext simulation data : %p\n", container.surfaceData, container.currentSimData, container.nextSimData);

	for (int y = 0; y < wH; y++) {
		for (int x = 0; x < wW; x++) {
			uint8_t* sd = (uint8_t*)container.currentSimData;

			if (x < 10 || x > wW - 10 || y < 10 || y > wH - 10) {
				sd[(x + (y * wH)) * 2] = 2;
				continue;
			}

			if (x < 40 && y < 40) {
				sd[(x + (y * wH)) * 2] = 1;
				sd[(x + (y * wH)) * 2 + 1] = 255;
			}
		}
	}

	container.size[0] = wW;
	container.size[1] = wH;

	return container;
}

void Loop(const Container& container) {
	bool isRunning{ true };

	int x, y;
	uint32_t mButtons;

	while (isRunning) {
		SDL_Event event;
		while (SDL_PollEvent(&event)) {
			switch (event.type) {
			case SDL_QUIT: {
				isRunning = false;
				break;
			}
			}
		}

		mButtons = SDL_GetMouseState(&x, &y);

		if ((mButtons & SDL_BUTTON_LMASK) != 0) {
			for (int ry = -10; ry < 10; ry++) {
				for (int rx = -10; rx < 10; rx++) {
					((uint8_t*)container.currentSimData)[(x + rx + ((y + ry) * container.surface->w)) * 2 + 1] = 255;
				}
			}
		}
		if ((mButtons & SDL_BUTTON_RMASK) != 0) {
			for (int ry = -10; ry < 10; ry++) {
				for (int rx = -10; rx < 10; rx++) {
					((uint8_t*)container.currentSimData)[(x + rx + ((y + ry) * container.surface->w)) * 2] = 2;
				}
			}
		}

		if ((mButtons & SDL_BUTTON_MMASK) != 0) {
			for (int ry = -10; ry < 10; ry++) {
				for (int rx = -10; rx < 10; rx++) {
					((uint8_t*)container.currentSimData)[(x + rx + ((y + ry) * container.surface->w)) * 2] = 0;
				}
			}
		}

		for (int i = 0; i < 8; i++) {
			ComputeSimulation(container.currentSimData, container.nextSimData, container.size, container.surface->w, container.surface->h);
			SDL_memcpy(container.currentSimData, container.nextSimData, (size_t)container.surface->w * container.surface->h * sizeof(uint16_t));
		}

		Draw(container.surfaceData, container.currentSimData, container.surface->w, container.surface->h);

		SDL_LockSurface(container.surface);
		SDL_memcpy(container.surface->pixels, container.surfaceData, (size_t)container.surface->h * container.surface->pitch);
		SDL_UnlockSurface(container.surface);

		SDL_Texture* buffer = SDL_CreateTextureFromSurface(container.renderer, container.surface);
		SDL_RenderCopy(container.renderer, buffer, nullptr, nullptr);
		SDL_RenderPresent(container.renderer);
		SDL_DestroyTexture(buffer);
		
	}
}

void Quit(const Container& container) {
	GPU_Free(container.nextSimData);
	GPU_Free(container.currentSimData);
	GPU_Free(container.surfaceData);
	SDL_FreeSurface(container.surface);
	SDL_DestroyRenderer(container.renderer);
	SDL_DestroyWindow(container.window);
	SDL_Quit();
}

int ShowErr(const char* msg) {
	MessageBoxA(nullptr, msg, "Error!", MB_OK);
	return EXIT_FAILURE;
}

int main() {
	Container container = Init("Fluid simulation GPU", 256, 256);

	try {
		Loop(container);
	}
	catch (const std::exception & e) {
		Quit(container);
		return ShowErr(e.what());
	}

	Quit(container);
	
	return 0;
}