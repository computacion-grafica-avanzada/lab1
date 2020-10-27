// ======================================================================== //
// Copyright 2018-2019 Ingo Wald                                            //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#pragma once

#define NUM_LIGHT_SAMPLES 1
#define NUM_PIXEL_SAMPLES 4

#define CUDA_CALL(x) do { if((x)!=cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)
#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)

/*! \namespace osc - Optix Siggraph Course */
namespace osc {

	static __forceinline__ __device__
		void* unpackPointer(uint32_t i0, uint32_t i1)
	{
		const uint64_t uptr = static_cast<uint64_t>(i0) << 32 | i1;
		void* ptr = reinterpret_cast<void*>(uptr);
		return ptr;
	}

	static __forceinline__ __device__
		void  packPointer(void* ptr, uint32_t& i0, uint32_t& i1)
	{
		const uint64_t uptr = reinterpret_cast<uint64_t>(ptr);
		i0 = uptr >> 32;
		i1 = uptr & 0x00000000ffffffff;
	}

	template<typename T>
	static __forceinline__ __device__ T* getPRD()
	{
		const uint32_t u0 = optixGetPayload_0();
		const uint32_t u1 = optixGetPayload_1();
		return reinterpret_cast<T*>(unpackPointer(u0, u1));
	}
	
	static __device__ __inline__ 
		vec3f reflect(vec3f v, vec3f n) 
	{
			return v - ((dot(v, n) / dot(n, n)) * n) * 2;
	}

	// sample hemisphere with cosine density
	static __device__ __inline__ 
		void sampleUnitHemisphere(const vec2f& sample, const vec3f& U, const vec3f& V, const vec3f& W, vec3f& point)
	{
		// https://github.com/nvpro-samples/optix_advanced_samples/blob/21465ae85c47f3c57371c77fd2aa3bac4adabcd4/src/optixProgressivePhotonMap/ppm_ppass.cu
		float phi = 2.0f * M_PI * sample.x;
		float r = sqrt(sample.y);
		float x = r * cos(phi);
		float y = r * sin(phi);
		float z = 1.0f - x * x - y * y;
		z = z > 0.0f ? sqrt(z) : 0.0f;

		point = x * U + y * V + z * W;
	}

	// Create ONB from normal.  Resulting W is parallel to normal
	static __device__ __inline__ 
		void create_onb(const vec3f& n, vec3f& U, vec3f& V, vec3f& W)
	{
		// https://github.com/nvpro-samples/optix_advanced_samples/blob/e4b6e03f5ad1239403d7990291604cd3eb12d814/src/device_include/helpers.h
		W = normalize(n);
		U = cross(W, vec3f(0.0f, 1.0f, 0.0f));

		if (fabs(U.x) < 0.001f && fabs(U.y) < 0.001f && fabs(U.z) < 0.001f)
			U = cross(W, vec3f(1.0f, 0.0f, 0.0f));

		U = normalize(U);
		V = cross(W, U);
	}

	static __device__ __inline__ 
		float max(const vec3f& vec) 
	{
		if ((vec.x > vec.y) && (vec.x > vec.z))
			return vec.x;

		if (vec.y > vec.z)
			return vec.y;

		return vec.z;
	}

	static __device__ __inline__
		int hash(vec3f position, vec3i gridSize, vec3f lowerBound) 
	{
		vec3f local = position - lowerBound;
		//vec3f G(
		//	local.x > 0.f ? floor(local.x / MAX_RADIUS) : 0.f,
		//	local.y > 0.f ? floor(local.y / MAX_RADIUS) : 0.f,
		//	local.z > 0.f ? floor(local.z / MAX_RADIUS) : 0.f
		//);
		vec3f G(
			fmaxf(0.f, floor(local.x / MAX_RADIUS)),
			fmaxf(0.f, floor(local.y / MAX_RADIUS)),
			fmaxf(0.f, floor(local.z / MAX_RADIUS))
		);
		return G.x + G.y * gridSize.x + G.z * gridSize.x * gridSize.y;
	}

	static __device__ __inline__
		int hash(vec3i position, vec3i gridSize)
	{
		return position.x + position.y * gridSize.x + position.z * gridSize.x * gridSize.y;
	}

	static __device__ __inline__
		int maximo(int a, int b) {
		return (a > b) ? a : b;
	}

	static __device__ __inline__
		int minimo(int a, int b) {
		return (a < b) ? a : b;
	}

} // ::osc
