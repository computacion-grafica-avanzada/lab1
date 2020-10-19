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

#include <optix_device.h>
#include <cuda_runtime.h>

#include "LaunchParams.h"
#include "gdt/random/random.h"

#include <cuda.h>

#define CUDA_CALL(x) do { if((x)!=cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)
#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)

using namespace osc;

#define NUM_LIGHT_SAMPLES 1
#define NUM_PIXEL_SAMPLES 16

#define NUM_PHOTON_SAMPLES 10000
#define MAX_DEPTH 10

namespace osc {

	typedef gdt::LCG<16> Random;

	/*! launch parameters in constant memory, filled in by optix upon
		optixLaunch (this gets filled in from the buffer we pass to
		optixLaunch) */
	extern "C" __constant__ LaunchParams optixLaunchParams;

	/*! per-ray data now captures random number generator, so programs
		can access RNG state */
	struct PRD {
		Random random;
		vec3f  pixelColor;
		int depth;
		int currentIor;
	};

	struct PhotonPRD {
		Random random;
		vec3f power;
		unsigned int depth;
	};



	static __forceinline__ __device__
		void* unpackPointer(uint32_t i0, uint32_t i1)
	{
		const uint64_t uptr = static_cast<uint64_t>(i0) << 32 | i1;
		void* ptr = reinterpret_cast<void*>(uptr);
		return ptr;
	}

	static __device__ __inline__ PhotonPRD getPhotonPRD()
	{
		PhotonPRD prd;
		prd.depth = optixGetPayload_0();
		return prd;
	}

	static __device__ __inline__ void setPhotonPRD(const PhotonPRD& prd)
	{
		optixSetPayload_0(prd.depth);
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

	static __device__ __inline__ vec3f reflect(vec3f v, vec3f n) {
		return ((dot(v, n) / dot(n, n)) * n) * 2 - v;
	}

	// sample hemisphere with cosine density
	static __device__ __inline__ void sampleUnitHemisphere(
		const vec2f& sample,
		const vec3f& U,
		const vec3f& V,
		const vec3f& W,
		vec3f& point)
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
	static __device__ __inline__ void create_onb(const vec3f& n, vec3f& U, vec3f& V, vec3f& W)
	{
		// https://github.com/nvpro-samples/optix_advanced_samples/blob/e4b6e03f5ad1239403d7990291604cd3eb12d814/src/device_include/helpers.h
		W = normalize(n);
		U = cross(W, vec3f(0.0f, 1.0f, 0.0f));

		if (fabs(U.x) < 0.001f && fabs(U.y) < 0.001f && fabs(U.z) < 0.001f)
			U = cross(W, vec3f(1.0f, 0.0f, 0.0f));

		U = normalize(U);
		V = cross(W, U);
	}

	static __device__ __inline__ float max(const vec3f& vec) {
		if ((vec.x > vec.y) && (vec.x > vec.z))
			return vec.x;
		
		if (vec.y > vec.z)
			return vec.y;
		
		return vec.z;
	}

	//------------------------------------------------------------------------------
	// closest hit and anyhit programs for radiance-type rays.
	//
	// Note eventually we will have to create one pair of those for each
	// ray type and each geometry type we want to render; but this
	// simple example doesn't use any actual geometries yet, so we only
	// create a single, dummy, set of them (we do have to have at least
	// one group of them to set up the SBT)
	//------------------------------------------------------------------------------

	extern "C" __global__ void __closesthit__shadow()
	{
		/* not going to be used ... */
	}

	extern "C" __global__ void __closesthit__photon()
	{
		// ------------------------------------------------------------------
		// gather some basic hit information
		// ------------------------------------------------------------------
		const TriangleMeshSBTData& sbtData = *(const TriangleMeshSBTData*)optixGetSbtDataPointer();
		const int   primID = optixGetPrimitiveIndex();
		const vec3i index = sbtData.index[primID];
		const int ix = optixGetLaunchIndex().x;

		// todo modulus with max number of photons
		int max_photons = sizeof(optixLaunchParams.halton) / sizeof(optixLaunchParams.halton[0]);

		PhotonPRD& prd = *(PhotonPRD*)getPRD<PhotonPRD>();

		const vec3f ray_orig = optixGetWorldRayOrigin();
		const vec3f ray_dir = optixGetWorldRayDirection();
		const float ray_t = optixGetRayTmax();

		vec3f hit_point = ray_orig + ray_t * ray_dir;

		printf("spec %f %f %f tran %f %f %f ior %f phong %f \n",
			sbtData.specular.x,
			sbtData.specular.y,
			sbtData.specular.z,
			sbtData.transmission.x,
			sbtData.transmission.y,
			sbtData.transmission.z,
			sbtData.ior,
			sbtData.phong
		);

		//for (int i = 0; i < hola.size(); i++) {
		//	printf("thrust %i\n", hola[i]);
		//}

		// ------------------------------------------------------------------
		// compute normal, using either shading normal (if avail), or
		// geometry normal (fallback)
		// ------------------------------------------------------------------
		const vec3f& A = sbtData.vertex[index.x];
		const vec3f& B = sbtData.vertex[index.y];
		const vec3f& C = sbtData.vertex[index.z];
		vec3f Ng = cross(B - A, C - A);

		if (dot(ray_dir, Ng) > 0.f) Ng = -Ng;
		Ng = normalize(Ng);
		
		printf("normal %f %f %f \n", Ng.x, Ng.y, Ng.z);

		//without random
		//PhotonPRD prd = getPhotonPRD();
		//prd.depth = 10;
		//setPhotonPRD(prd);
		printf("hit %i \n", prd.depth);

		int rand_index = (ix * prd.depth) % max_photons;
		float coin = optixLaunchParams.halton[rand_index].x;

		// diffuse component is color for now
		float Pd = max(sbtData.color * prd.power) / max(prd.power);
		printf("color %f %f %f maximo %f prob difuso %f mult %f %f %f\n", sbtData.color.x, sbtData.color.y, sbtData.color.z, max(sbtData.color), Pd, (sbtData.color * prd.power).x, (sbtData.color * prd.power).y, (sbtData.color * prd.power).z);

		prd.depth += 1;
		if (coin <= Pd) {
			//diffuse
			
			// avoid first diffuse hit
			if (prd.depth > 1) {
				PhotonPrint pp = { hit_point, ray_dir, prd.power };
				optixLaunchParams.photons[ix * MAX_DEPTH + prd.depth - 2] = pp;
			}

			if (prd.depth <= MAX_DEPTH) {
				uint32_t u0, u1;
				packPointer(&prd, u0, u1);

				// obtain random direction
				vec3f U, V, W, direction;
				create_onb(Ng, U, V, W);

				sampleUnitHemisphere(optixLaunchParams.halton[rand_index].y, U, V, W, direction);
			
				printf("direction %f %f %f \n", direction.x, direction.y, direction.z);

				prd.power = (prd.power * sbtData.color) / Pd;

				optixTrace(
					optixLaunchParams.traversable,
					hit_point,
					direction,
					0.f,							// tmin
					1e20f,							// tmax
					0.0f,							// rayTime
					OptixVisibilityMask(255),
					OPTIX_RAY_FLAG_DISABLE_ANYHIT,	// OPTIX_RAY_FLAG_NONE,
					PHOTON_RAY_TYPE,				// SBT offset
					RAY_TYPE_COUNT,					// SBT stride
					PHOTON_RAY_TYPE,				// missSBTIndex 
					//prd.depth						// reinterpret_cast<unsigned int&>(prd.depth)
					u0, u1
				);
			}
		}
		// falta if de speculares
		else {
			// absorption check if need to store diffuse photons
		}
	
		
		// ruleta rusa
		// en el caso difuso si depth es 0 no se guarda marca
		// chequear depth+1 menor a MAX
		// tirar rayos acorde a reflexion difusa, specular, transmision o nada si se absorbe

	}

	extern "C" __global__ void __closesthit__radiance()
	{
		// direct
		//	rayos de sombra -> visibilidad de la luz * color difuso?
		// specular and glossy
		//	reflexiones y transmisiones como en ray tracing
		// caustics
		//	radiance estimate caustics map
		// multiple diffuse reflections
		//	primer paso, sacar radiancia del hit nomas
		//	trazar reflexiones difusas y ver radiancia en mapa de fotones para cada uno

		const TriangleMeshSBTData& sbtData = *(const TriangleMeshSBTData*)optixGetSbtDataPointer();
		PRD& prd = *getPRD<PRD>();

		// ------------------------------------------------------------------
		// gather some basic hit information
		// ------------------------------------------------------------------
		const int   primID = optixGetPrimitiveIndex();
		const vec3i index = sbtData.index[primID];
		const float u = optixGetTriangleBarycentrics().x;
		const float v = optixGetTriangleBarycentrics().y;

		// ------------------------------------------------------------------
		// compute normal, using either shading normal (if avail), or
		// geometry normal (fallback)
		// ------------------------------------------------------------------
		const vec3f& A = sbtData.vertex[index.x];
		const vec3f& B = sbtData.vertex[index.y];
		const vec3f& C = sbtData.vertex[index.z];
		vec3f Ng = cross(B - A, C - A);

		// ------------------------------------------------------------------
		// face-forward and normalize normals
		// ------------------------------------------------------------------
		const vec3f rayDir = optixGetWorldRayDirection();

		float cosi = dot(rayDir, Ng);
		if (cosi > 0.f) Ng = -Ng;
		Ng = normalize(Ng);

		// start with some ambient term
		//vec3f pixelColor = (0.1f + 0.2f * fabsf(dot(Ng, rayDir))) * sbtData.color;
		vec3f pixelColor = 0;
		//printf("pixel1 %f %f %f pixel2 %f %f %f\n", pixelColor.x, pixelColor.y, pixelColor.z, pixelColor1.x, pixelColor1.y, pixelColor1.z);

		// ------------------------------------------------------------------
		// compute shadow
		// ------------------------------------------------------------------
		const vec3f surfPos
			= (1.f - u - v) * sbtData.vertex[index.x]
			+ u * sbtData.vertex[index.y]
			+ v * sbtData.vertex[index.z];

		const vec3f ray_orig = optixGetWorldRayOrigin();
		const vec3f ray_dir = optixGetWorldRayDirection();
		const float ray_t = optixGetRayTmax();

		vec3f hit_point = ray_orig + ray_t * ray_dir;
		//printf("hit %f %f %f surf %f %f %f \n", hit_point.x, hit_point.y, hit_point.z, surfPos.x, surfPos.y, surfPos.z);

		//for each light todo
		const int numLightSamples = NUM_LIGHT_SAMPLES;
		for (int lightSampleID = 0; lightSampleID < numLightSamples; lightSampleID++) {

			// produce random light sample
			const vec3f lightPos = optixLaunchParams.light.origin;
			//+ prd.random() * optixLaunchParams.light.du
			//+ prd.random() * optixLaunchParams.light.dv;
			vec3f lightDir = lightPos - surfPos;
			float lightDist = length(lightDir);
			lightDir = normalize(lightDir);

			// trace shadow ray:
			const float NdotL = dot(lightDir, Ng);
			if (NdotL >= 0.f) {
				vec3f lightVisibility = 1.f;
				
				// the values we store the PRD pointer in:
				uint32_t u0, u1;
				packPointer(&lightVisibility, u0, u1);

				optixTrace(optixLaunchParams.traversable,
					hit_point,
					lightDir,
					1e-3f,      // tmin
					lightDist * (1.f - 1e-3f),  // tmax
					0.0f,       // rayTime
					OptixVisibilityMask(255),
					// For shadow rays: skip any/closest hit shaders and terminate on first
					// intersection with anything. The miss shader is used to mark if the
					// light was visible.
					/*OPTIX_RAY_FLAG_DISABLE_ANYHIT
					| OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT
					|*/ 
					OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
					SHADOW_RAY_TYPE,            // SBT offset
					RAY_TYPE_COUNT,             // SBT stride
					SHADOW_RAY_TYPE,            // missSBTIndex 
					u0, u1);

				float atenuation = 1 / (lightDist * lightDist);
				//float atenuation = 1;
				vec3f base = lightVisibility * atenuation * optixLaunchParams.light.photonPower;

				vec3f reflected = ((dot(lightDir, Ng) / dot(Ng,Ng)) * Ng) * 2 - lightDir;
				//printf("light %f %f %f reflect %f %f %f normal %f %f %f\n", lightDir.x, lightDir.y, lightDir.z, reflected.x, reflected.y, reflected.z, Ng.x,Ng.y,Ng.z);
				float a = dot(reflected, -rayDir);
				float RdotV = a < 0 ? 0 : a;

				pixelColor += base * sbtData.color * (NdotL / numLightSamples);
				pixelColor += base * sbtData.specular * pow(RdotV, sbtData.phong) / numLightSamples;
			}
		}

		///*Regresar si la prof. es excesiva */
		if (prd.depth < MAX_DEPTH) {
			prd.depth += 1;

			// reflection component
			if (sbtData.specular.x > 0 || sbtData.specular.y > 0 || sbtData.specular.z > 0) {
				vec3f reflectDir = reflect(-rayDir, Ng); //rayo en la direccion de reflexion desde punto;

				uint32_t u0, u1;
				packPointer(&prd, u0, u1);

				optixTrace(optixLaunchParams.traversable,
					hit_point,
					reflectDir,
					1.e-4f,    // tmin
					1e20f,  // tmax
					0.0f,   // rayTime
					OptixVisibilityMask(255),
					OPTIX_RAY_FLAG_DISABLE_ANYHIT,//OPTIX_RAY_FLAG_NONE,
					RADIANCE_RAY_TYPE,            // SBT offset
					RAY_TYPE_COUNT,               // SBT stride
					RADIANCE_RAY_TYPE,            // missSBTIndex 
					u0, u1
				);

				pixelColor += sbtData.specular * prd.pixelColor;
			}

			// refraction component
			if (sbtData.transmission.x > 0 || sbtData.transmission.y > 0 || sbtData.transmission.z > 0) {
				//printf("hit algo\n");
				float nit;
				// ray comes from outside surface
				if (cosi < 0) {
					nit = prd.currentIor / sbtData.ior;
					cosi = -cosi;
					prd.currentIor = sbtData.ior;
					//printf("holas");
					//Ng = -Ng;
				}
				// ray comes from inside surface
				else {
					// we assume that when it leaves the surface, it goes into the air
					nit = sbtData.ior;
				}

				float testrit = nit * nit * (1 - cosi * cosi);

				if (testrit <= 1) {
					vec3f refractionDir = nit * rayDir + (nit * cosi - sqrtf(1 - testrit)) * Ng;
					
					uint32_t u0, u1;
					packPointer(&prd, u0, u1);

					optixTrace(optixLaunchParams.traversable,
						surfPos,
						refractionDir,
						1e-4f,    // tmin
						1e20f,  // tmax
						0.0f,   // rayTime
						OptixVisibilityMask(255),
						OPTIX_RAY_FLAG_DISABLE_ANYHIT,//OPTIX_RAY_FLAG_NONE,
						RADIANCE_RAY_TYPE,            // SBT offset
						RAY_TYPE_COUNT,               // SBT stride
						RADIANCE_RAY_TYPE,            // missSBTIndex 
						u0, u1
					);
					pixelColor = sbtData.transmission * prd.pixelColor;
				}
			}
		}
		//pixelColor = lightVisibility;
		prd.pixelColor = pixelColor;
		//printf("output %f %f %f, pixel %f %f %f\n", 
		//	prd.pixelColor.x, prd.pixelColor.y, prd.pixelColor.z,
		//	pixelColor.x, pixelColor.y, pixelColor.z);
	}

	extern "C" __global__ void __anyhit__radiance()
	{ /*! for this simple example, this will remain empty */
	}

	extern "C" __global__ void __anyhit__shadow()
	{ /*! not going to be used */
		const TriangleMeshSBTData& sbtData = *(const TriangleMeshSBTData*)optixGetSbtDataPointer();
		vec3f& prd = *getPRD<vec3f>();
		prd *= sbtData.transmission;
		//if (sbtData.transmission != vec3f(0))
		//	printf("%f %f %f\n",prd.x, prd.y, prd.z);
	}

	extern "C" __global__ void __anyhit__photon()
	{ /*! not going to be used */
	}

	//------------------------------------------------------------------------------
	// miss program that gets called for any ray that did not have a
	// valid intersection
	//
	// as with the anyhit/closest hit programs, in this example we only
	// need to have _some_ dummy function to set up a valid SBT
	// ------------------------------------------------------------------------------

	extern "C" __global__ void __miss__radiance()
	{
		PRD& prd = *getPRD<PRD>();
		// set to constant white as background color
		prd.pixelColor = vec3f(1,1,0);
	}

	extern "C" __global__ void __miss__shadow()
	{
		// we didn't hit anything, so the light is visible
		vec3f& prd = *(vec3f*)getPRD<vec3f>();
		prd = vec3f(1.f);
	}

	extern "C" __global__ void __miss__photon()
	{
		PhotonPRD& prd = *(PhotonPRD*)getPRD<PhotonPRD>();
		//prd.depth = prd.random() * 10;

		//without random
		//PhotonPRD prd = getPhotonPRD();
		//prd.depth = 100;
		//setPhotonPRD(prd);
		printf("miss %i \n", prd.depth);
	}

	extern "C" __global__ void __raygen__renderPhoton()
	{
		// compute a test pattern based on pixel ID
		const int ix = optixGetLaunchIndex().x;

		Random ran;
		ran.init(ix,ix*optixLaunchParams.frame.size.x);

		printf("\n random 1: %f 2: %f 3: %f ix: %d hax: %f hay: %f\n", ran(), ran(), ran(), ix, optixLaunchParams.halton[ix].x, optixLaunchParams.halton[ix].y);
		
		PhotonPRD prd;
		prd.random.init(ix, ix * optixLaunchParams.frame.size.x);
		prd.depth = 0;
		prd.power = optixLaunchParams.light.photonPower;
		printf("power photon %f %f %f \n", prd.power.x, prd.power.y, prd.power.z);

		uint32_t u0, u1;
		packPointer(&prd, u0, u1);

		// obtain random direction
		vec3f U, V, W, direction;
		create_onb(optixLaunchParams.light.normal, U, V, W);
		sampleUnitHemisphere(optixLaunchParams.halton[ix], U, V, W, direction);
				
		optixTrace(
			optixLaunchParams.traversable,
			optixLaunchParams.light.origin,
			direction,
			0.f,							// tmin
			1e20f,							// tmax
			0.0f,							// rayTime
			OptixVisibilityMask(255),
			OPTIX_RAY_FLAG_DISABLE_ANYHIT,	// OPTIX_RAY_FLAG_NONE,
			PHOTON_RAY_TYPE,				// SBT offset
			RAY_TYPE_COUNT,					// SBT stride
			PHOTON_RAY_TYPE,				// missSBTIndex 
			//prd.depth						// reinterpret_cast<unsigned int&>(prd.depth)
			u0,u1
		);

		printf("profundidad %i\n", prd.depth);
	}

	//------------------------------------------------------------------------------
	// ray gen program - the actual rendering happens in here
	//------------------------------------------------------------------------------
	extern "C" __global__ void __raygen__renderFrame()
	{
		// compute a test pattern based on pixel ID
		const int ix = optixGetLaunchIndex().x;
		const int iy = optixGetLaunchIndex().y;
		const int accumID = optixLaunchParams.frame.accumID;
		const auto& camera = optixLaunchParams.camera;

		PRD prd;
		prd.random.init(ix + accumID * optixLaunchParams.frame.size.x, iy + accumID * optixLaunchParams.frame.size.y);
		prd.pixelColor = vec3f(0.f);
		prd.depth = 0;
		prd.currentIor = 1;

		// the values we store the PRD pointer in:
		uint32_t u0, u1;
		packPointer(&prd, u0, u1);

		int numPixelSamples = NUM_PIXEL_SAMPLES;

		vec3f pixelColor = 0.f;
		for (int sampleID = 0; sampleID < numPixelSamples; sampleID++) {
			// normalized screen plane position, in [0,1]^2
			const vec2f screen(vec2f(ix + prd.random(), iy + prd.random()) / vec2f(optixLaunchParams.frame.size));

			// generate ray direction
			vec3f rayDir = normalize(camera.direction 
				+ (screen.x - 0.5f) * camera.horizontal
				+ (screen.y - 0.5f) * camera.vertical);

			optixTrace(optixLaunchParams.traversable,
				camera.position,
				rayDir,
				1e-4f,    // tmin
				1e20f,  // tmax
				0.0f,   // rayTime
				OptixVisibilityMask(255),
				OPTIX_RAY_FLAG_DISABLE_ANYHIT,//OPTIX_RAY_FLAG_NONE,
				RADIANCE_RAY_TYPE,            // SBT offset
				RAY_TYPE_COUNT,               // SBT stride
				RADIANCE_RAY_TYPE,            // missSBTIndex 
				u0, u1);
			pixelColor += prd.pixelColor;
		}
		
		const int r = int(255.99f * min(pixelColor.x / numPixelSamples, 1.f));
		const int g = int(255.99f * min(pixelColor.y / numPixelSamples, 1.f));
		const int b = int(255.99f * min(pixelColor.z / numPixelSamples, 1.f));

		// convert to 32-bit rgba value (we explicitly set alpha to 0xff
		// to make stb_image_write happy ...
		const uint32_t rgba = 0xff000000
			| (r << 0) | (g << 8) | (b << 16);

		// and write to frame buffer ...
		const uint32_t fbIndex = ix + iy * optixLaunchParams.frame.size.x;
		optixLaunchParams.frame.colorBuffer[fbIndex] = rgba;
	}

} // ::osc
