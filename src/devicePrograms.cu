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
#include "Utils.h"

using namespace osc;

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

	// todo check disk and use n photons
	static __device__ void nearest_photons(vec3f point, float max_r, int gridId) {
		//float* distances = new float[optixLaunchParams.mapSize];
		//cudaMallocManaged(&distances, optixLaunchParams.mapSize * sizeof(float));
		int count = 0;
		float sq_r = max_r * max_r;
		//printf("max size %i\n", optixLaunchParams.mapSize);
		for (int i = 0; i < optixLaunchParams.mapSize; i++) {
			vec3f diff = optixLaunchParams.photonMap[i].position - point;
			float sq_dist = dot(diff,diff);
			//printf("distancia %f\n", dist);
			//distances[i] = dist;
			if (sq_dist <= sq_r) {
				optixLaunchParams.nearestPhotons[gridId + count] = optixLaunchParams.photonMap[i];
				count++;
			}
			if (count >= 100)
				break;
		}
		//PhotonPrint* res = new PhotonPrint[count];
		//(*qw) = new PhotonPrint[count];
		////cudaMallocManaged(&qw, count * sizeof(PhotonPrint));
		//int current = 0;
		//for (int i = 0; i < optixLaunchParams.mapSize; i++) {
		//	if (distances[i] <= max_r) {
		//		(*qw)[current] = optixLaunchParams.photonMap[i];
		//		current++;
		//	}
		//}
		// for sobre todos los fotones para contar cuantos hay que cumplen
		// crear array de ese largo
		// for sobre todos y agregarlos
		//printf("nearest %f %f %f count %i\n", point.x, point.y, point.z, count);
		//delete [] distances;
		//cudaFree(distances);

		//return res;
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
		//float* we = new float[3];
		//delete[] we;
		//float we[3];
		//we[0] = 0.4f;
		//we[1] = 2.f;
		//we[2] = 0.31f;
		//printf("%f %f %f", we[0], we[1], we[2]);
		const int ix = optixGetLaunchIndex().x;
		const int iy = optixGetLaunchIndex().y;
		int gridId = ix * 800 * 100 + iy * 100;
		nearest_photons(hit_point, 0.5f, gridId);
		//printf("wer %f %f %f\n", qw[0].position.x, qw[0].position.y, qw[0].position.z);

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

				vec3f reflected = ((dot(lightDir, Ng) / dot(Ng, Ng)) * Ng) * 2 - lightDir;
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

				PRD reflection;
				reflection.pixelColor = vec3f(0.f);
				reflection.depth = prd.depth;
				reflection.currentIor = prd.currentIor;

				uint32_t u0, u1;
				packPointer(&reflection, u0, u1);

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

				pixelColor += sbtData.specular * reflection.pixelColor;
			}

			// refraction component
			if (sbtData.transmission.x > 0 || sbtData.transmission.y > 0 || sbtData.transmission.z > 0) {
				//printf("hit algo\n");
				float nit;
				PRD refraction;
				refraction.pixelColor = vec3f(0.f);
				refraction.depth = prd.depth;
				refraction.currentIor = prd.currentIor;

				// ray comes from outside surface
				if (cosi > 0) {
					// we assume that when it leaves the surface, it goes into the air
					nit = sbtData.ior;
					cosi = -cosi;
					refraction.currentIor = sbtData.ior;
					//printf("holas");
					//Ng = -Ng;
				}
				// ray comes from inside surface
				else {
					nit = prd.currentIor / sbtData.ior;
				}

				float testrit = nit * nit * (1 - cosi * cosi);

				if (testrit <= 1) {
					vec3f refractionDir = nit * rayDir + (nit * cosi - sqrtf(1 - testrit)) * Ng;

					uint32_t u0, u1;
					packPointer(&refraction, u0, u1);

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
					pixelColor += sbtData.transmission * refraction.pixelColor;
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
		prd.pixelColor = vec3f(1, 1, 0);
	}

	extern "C" __global__ void __miss__shadow()
	{
		// we didn't hit anything, so the light is visible
		vec3f& prd = *(vec3f*)getPRD<vec3f>();
		prd = vec3f(1.f);
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

		int numPixelSamples = 4; //NUM_PIXEL_SAMPLES;


		vec3f pixelColor = 0.f;
		for (int sampleID = 0; sampleID < numPixelSamples; sampleID++) {
			prd.pixelColor = vec3f(0.f);
			// normalized screen plane position, in [0,1]^2
			const vec2f screen(vec2f(ix + prd.random(), iy + prd.random()) / vec2f(optixLaunchParams.frame.size));
			//const vec2f screen(vec2f(ix, iy));

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
			//if (ix == 600 && iy == 400)
			//{
			//	printf("\SCREEN: %i %i", optixLaunchParams.frame.size.x, optixLaunchParams.frame.size.y);
			//	printf("\nROJO: %f   SAMPLE: %i", prd.pixelColor.x, sampleID);
			//	printf("\nVERDE: %f   SAMPLE: %i", prd.pixelColor.y, sampleID);
			//	printf("\nAZUL: %f   SAMPLE: %i \n", prd.pixelColor.z, sampleID);
			//}
		}

		const int r = int(255.99f * min(pixelColor.x / numPixelSamples, 1.f));
		const int g = int(255.99f * min(pixelColor.y / numPixelSamples, 1.f));
		const int b = int(255.99f * min(pixelColor.z / numPixelSamples, 1.f));

		//if (ix == 600 && iy == 400)
		//{
		//	printf("\nROJO FINAL: %f", pixelColor.x / numPixelSamples);
		//	printf("\nVERDE FINAL: %f", pixelColor.y / numPixelSamples);
		//	printf("\nAZUL FINAL: %f \n\n", pixelColor.z / numPixelSamples);
		//}


		// convert to 32-bit rgba value (we explicitly set alpha to 0xff
		// to make stb_image_write happy ...
		const uint32_t rgba = 0xff000000
			| (r << 0) | (g << 8) | (b << 16);

		// and write to frame buffer ...
		const uint32_t fbIndex = ix + iy * optixLaunchParams.frame.size.x;
		optixLaunchParams.frame.colorBuffer[fbIndex] = rgba;
	}

} // ::osc
