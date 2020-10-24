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

#define ALPHA 0.918
#define BETA 1.953

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
		float currentIor;
	};

	struct PhotonPRD {
		Random random;
		vec3f power;
		unsigned int depth;
	};

	// todo check disk and use n photons
	static __device__ vec3f nearest_photons(vec3f point, float max_r) {
		//float* distances = new float[optixLaunchParams.mapSize];
		//cudaMallocManaged(&distances, optixLaunchParams.mapSize * sizeof(float));
		int count = 0;
		vec3f totalPower = 0;
		float sq_r = max_r * max_r;

		//float act_r = 0;
		//for (int i = 0; i < optixLaunchParams.mapSize; i++) {
		//	vec3f diff = optixLaunchParams.photonMap[i].position - point;
		//	float sq_dist = dot(diff, diff);


		//	//printf("distancia %f\n", dist);
		//	//distances[i] = dist;
		//	if (sq_dist <= sq_r) {
		//		if (sq_dist > act_r) act_r = sq_dist;
		//		//totalPower += optixLaunchParams.photonMap[i].power; // *w_pg;
		//		//count++;
		//		//printf("count %i sq_dist %f sq_r %f\n", count, sq_dist, sq_r);
		//	}
		//	// maximum of 10 neighbors
		//	//if (count >= MAX_DEPTH)
		//	//	break;
		//}

		float denominator = 1 - expf(-BETA);

		//printf("max size %i\n", optixLaunchParams.mapSize);
		for (int i = 0; i < optixLaunchParams.mapSize; i++) {
			vec3f diff = optixLaunchParams.photonMap[i].position - point;
			float sq_dist = dot(diff,diff);
			
			float numerator = 1 - expf(-BETA * (sq_dist / 2 * sq_r));
			float w_pg = ALPHA * (1 - numerator/denominator);
			
			//printf("distancia %f\n", dist);
			//distances[i] = dist;
			if (sq_dist <= sq_r) {
				totalPower += optixLaunchParams.photonMap[i].power *w_pg;
				count++;
				//printf("count %i sq_dist %f sq_r %f\n", count, sq_dist, sq_r);
			}
			// maximum of 10 neighbors
			//if (count >= MAX_DEPTH)
			//	break;
		}
		//printf("count %i power %f %f %f\n", count, totalPower.x, totalPower.y, totalPower.z);
		return totalPower; // / count;
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
		//printf("final count %i\n", count);
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

		PRD& prd = *getPRD<PRD>();
		const TriangleMeshSBTData& sbtData = *(const TriangleMeshSBTData*)optixGetSbtDataPointer();
		const int ix = optixGetLaunchIndex().x;
		const int iy = optixGetLaunchIndex().y;

		// ------------------------------------------------------------------
		// gather some basic hit information
		// ------------------------------------------------------------------
		const int  primID = optixGetPrimitiveIndex();
		const vec3i index = sbtData.index[primID];

		// ------------------------------------------------------------------
		// compute normal, using either shading normal (if avail), or
		// geometry normal (fallback)
		// ------------------------------------------------------------------
		const vec3f& A = sbtData.vertex[index.x];
		const vec3f& B = sbtData.vertex[index.y];
		const vec3f& C = sbtData.vertex[index.z];
		vec3f Ng = cross(B - A, C - A);

		// ------------------------------------------------------------------
		// compute ray intersection point
		// ------------------------------------------------------------------
		const vec3f rayOrig = optixGetWorldRayOrigin();
		const vec3f rayDir = optixGetWorldRayDirection();
		const float rayT = optixGetRayTmax();
		vec3f hitPoint = rayOrig + rayDir * rayT;

		// ------------------------------------------------------------------
		// face-forward and normalize normals
		// ------------------------------------------------------------------
		float cosi = dot(rayDir, Ng);
		if (cosi > 0.f) Ng = -Ng;
		Ng = normalize(Ng);

		// start with some ambient term
		vec3f pixelColor(0.f);

		// ------------------------------------------------------------------
		// compute multiple diffuse
		// ------------------------------------------------------------------
		//if (sbtData.color != vec3f(0)) {
		//	float radius = 0.25f;
		//	vec3f totalPower = nearest_photons(hitPoint, radius);

		//	vec3f brdf = sbtData.color / M_PI;
		//	//printf("brdf %f %f %f", brdf.x, brdf.y, brdf.z);

		//	//vec3f totalPower = 0;
		//	//for (int i = 0; i < nn; i++) {
		//	//	totalPower += optixLaunchParams.nearestPhotons[gridId + i].power;
		//	//}


		//	float delta_a = M_PI * radius * radius;
		//	//printf("%f brdf %f %f %f delta %f %f %f\n", delta_a, brdf.x, brdf.y, brdf.z, (brdf/delta_a).x, (brdf / delta_a).y, (brdf / delta_a).z);
		//	//pixelColor += (brdf / delta_a) * totalPower;
		//	pixelColor += totalPower * brdf;
		//	//pixelColor += optixLaunchParams.nearestPhotons[gridId].power;
		//}

		// ------------------------------------------------------------------
		// compute shadow
		// ------------------------------------------------------------------
		//for each light todo
		for (int lightSampleID = 0; lightSampleID < NUM_LIGHT_SAMPLES; lightSampleID++) {

			// produce random light sample
			const vec3f lightPos = optixLaunchParams.light.origin;
			//+ prd.random() * optixLaunchParams.light.du
			//+ prd.random() * optixLaunchParams.light.dv;
			vec3f lightDir = lightPos - hitPoint; // rayo hacia la luz
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
					hitPoint,
					lightDir,
					1.e-4f,						// tmin
					lightDist * (1.f - 1e-3f),  // tmax
					0.0f,						// rayTime
					OptixVisibilityMask(255),
					// For shadow rays: skip closest hit shaders and terminate on first
					// intersection with an opaque object. The miss shader is used to mark if the
					// light was visible.
					OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
					SHADOW_RAY_TYPE,            // SBT offset
					RAY_TYPE_COUNT,             // SBT stride
					SHADOW_RAY_TYPE,            // missSBTIndex 
					u0, u1);

				float atenuation = 1 / (lightDist * lightDist);
				vec3f base = lightVisibility * atenuation * optixLaunchParams.light.photonPower;

				vec3f reflectDir = reflect(-lightDir, Ng);
				float RdotV = fmaxf(0.f, dot(reflectDir, -rayDir));

				pixelColor += base * sbtData.color * (NdotL / NUM_LIGHT_SAMPLES);
				pixelColor += base * sbtData.specular * pow(RdotV, sbtData.phong) / NUM_LIGHT_SAMPLES;
			}
		}

		///*Regresar si la prof. es excesiva */
		if (prd.depth < MAX_DEPTH) {
			prd.depth += 1;

			//// reflection component
			if (sbtData.specular != vec3f(0)) {
				vec3f reflectDir = reflect(rayDir, Ng); //rayo en la direccion de reflexion desde punto;

				PRD reflection;
				reflection.pixelColor = vec3f(0.f);
				reflection.depth = prd.depth;
				reflection.currentIor = prd.currentIor;

				uint32_t u0, u1;
				packPointer(&reflection, u0, u1);

				optixTrace(optixLaunchParams.traversable,
					hitPoint,
					reflectDir,
					1.e-4f, // tmin
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
			if (sbtData.transmission != vec3f(0)) {
				float nit;
				PRD refraction;
				refraction.pixelColor = vec3f(0.f);
				refraction.depth = prd.depth;
				refraction.currentIor = prd.currentIor;

				// ray comes from outside surface
				if (cosi < 0) {
					// we assume that when it leaves the surface, it goes into the air
					nit = prd.currentIor / sbtData.ior;
					cosi = -cosi;
					refraction.currentIor = sbtData.ior;
					//printf("holas");
					//Ng = -Ng;
				}
				// ray comes from inside surface
				else {
					nit = sbtData.ior;
					refraction.currentIor = 1;
				}

				float testrit = nit * nit * (1 - cosi * cosi);

				if (testrit <= 1) {
					vec3f refractionDir = nit * rayDir + (nit * cosi - sqrtf(1 - testrit)) * Ng;

					uint32_t u0, u1;
					packPointer(&refraction, u0, u1);

					optixTrace(optixLaunchParams.traversable,
						hitPoint,
						refractionDir,
						1e-4f,  // tmin
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
		//if (pixelColor.x > 1 || pixelColor.y > 1 || pixelColor.z > 1)
		//	printf("%f %f %f\n", pixelColor.x, pixelColor.y, pixelColor.z);
		prd.pixelColor = pixelColor;
	}

	extern "C" __global__ void __anyhit__radiance()
	{ /*! for this simple example, this will remain empty */
	}

	extern "C" __global__ void __anyhit__shadow()
	{
		const TriangleMeshSBTData& sbtData = *(const TriangleMeshSBTData*)optixGetSbtDataPointer();
		vec3f& prd = *getPRD<vec3f>();
		prd *= sbtData.transmission;
		// object is not opaque
		if (sbtData.transmission != vec3f(0)) optixIgnoreIntersection();
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
		prd.pixelColor = vec3f(0.5f);
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

		// the values we store the PRD pointer in:
		uint32_t u0, u1;
		packPointer(&prd, u0, u1);

		//int numPixelSamples = 1; // 4; //NUM_PIXEL_SAMPLES;

		int antialiasingLevel = 10;
		int numPixelSamples = antialiasingLevel * antialiasingLevel;
		float cellWidth = 1.f / antialiasingLevel;
		float increment = cellWidth * 0.5;

		//if (ix == 600 && iy == 400) printf("#### start\n");
		vec3f pixelColor = 0.f;
		for (int ai = 0; ai < antialiasingLevel; ai++) {
			for (int aj = 0; aj < antialiasingLevel; aj++) {
				prd.pixelColor = vec3f(0.f);
				prd.depth = 0;
				prd.currentIor = 1;
				
				vec2f sample(
					 ix + ai * cellWidth + increment, 
					 iy + aj * cellWidth + increment
				);
				const vec2f screen(sample / vec2f(optixLaunchParams.frame.size));
				//const vec2f screen(vec2f(ix + prd.random(), iy + prd.random()) / vec2f(optixLaunchParams.frame.size));
				 
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

				 //if (ix == 600 && iy == 400) printf("%f %f %f %i\n", prd.pixelColor.x, prd.pixelColor.y, prd.pixelColor.z, prd.depth);
				 pixelColor += prd.pixelColor;

			}
		}
		//if (ix == 600 && iy == 400) printf("#### end\n");
		//}

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
