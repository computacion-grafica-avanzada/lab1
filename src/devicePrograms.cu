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
	//static __device__ vec3f nearest_photons(vec3f point, float max_r) {
	//	vec3f totalPower = 0;
	//	float sq_r = max_r * max_r;

	//	//float denominator = 1 - expf(-BETA);

	//	for (int i = 0; i < optixLaunchParams.mapSize; i++) {
	//		vec3f dir = optixLaunchParams.photonMap[i].position - point;
	//		float sq_dist = dot(dir,dir);
	//		
	//		//float numerator = 1 - expf(-BETA * (sq_dist / 2 * sq_r));
	//		//float w_pg = ALPHA * (1 - numerator/denominator);
	//		
	//		//printf("hola %f %f\n", sq_dist, sq_r);
	//		if (sq_dist <= sq_r) {
	//			totalPower += optixLaunchParams.photonMap[i].power; // *w_pg;
	//		}
	//	}
	//	return totalPower;
	//}

	//static __device__ vec3f nearest_photons_hash(vec3f point) {
	//	vec3f totalPower(0.f);
	//	vec3f radius(MAX_RADIUS);
	//	float sq_r = MAX_RADIUS * MAX_RADIUS;

	//	vec3f local = point - optixLaunchParams.lowerBound;
	//	vec3f minPoint = (local - radius) / radius;
	//	vec3i from(
	//		fmaxf(0.f, floor(minPoint.x)),
	//		fmaxf(0.f, floor(minPoint.y)),
	//		fmaxf(0.f, floor(minPoint.z))
	//	);
	//	vec3f maxPoint = (local + radius) / radius;
	//	vec3i to(
	//		fminf(optixLaunchParams.gridSize.x-1, floor(maxPoint.x)),
	//		fminf(optixLaunchParams.gridSize.y-1, floor(maxPoint.y)),
	//		fminf(optixLaunchParams.gridSize.z-1, floor(maxPoint.z))
	//	);
	//	//printf("min %i %i %i max %i %i %i\n", from.x, from.y, from.z, to.x, to.y, to.z);

	//	for (int x = from.x; x <= to.x; x++) {
	//		for (int y = from.y; y <= to.y; y++) {
	//			for (int z = from.z; z <= to.z; z++) {
	//				int hashId = hash(
	//					vec3f(x, y, z), 
	//					optixLaunchParams.gridSize
	//				);
	//				
	//				if (optixLaunchParams.pmCount[hashId] > 0) {
	//					vec3f dir = optixLaunchParams.pm[hashId].position - point;
	//					float sq_dist = dot(dir, dir);
	//					if (sq_dist <= sq_r)
	//						totalPower += optixLaunchParams.pm[hashId].power * optixLaunchParams.pmCount[hashId];
	//				}
	//			}
	//		}

	//	}
	//	return totalPower;
	//}

	static __device__ vec3f nearest_photons_hash_list(vec3f point, float radius_f) {
		vec3f totalPower(0.f);
		vec3f radius(optixLaunchParams.maxRadius);
		float sq_r = radius_f * radius_f;

		vec3f local = point - optixLaunchParams.lowerBound;
		// find min 3D grid index
		vec3f minPoint = (local - radius) / radius;
		vec3i from(
			maximo(0, (int)minPoint.x),
			maximo(0, (int)minPoint.y),
			maximo(0, (int)minPoint.z)
		);
		// find max 3D grid index
		vec3f maxPoint = (local + radius) / radius;
		vec3i to(
			minimo(optixLaunchParams.gridSize.x - 1, (int)maxPoint.x),
			minimo(optixLaunchParams.gridSize.y - 1, (int)maxPoint.y),
			minimo(optixLaunchParams.gridSize.z - 1, (int)maxPoint.z)
		);

		int count = 0;
		for (int x = from.x; x <= to.x; x++) {
			for (int y = from.y; y <= to.y; y++) {
				for (int z = from.z; z <= to.z; z++) {
					int hashId = hash(
						vec3i(x, y, z),
						optixLaunchParams.gridSize
					);
					int start = optixLaunchParams.pmStarts[hashId];
					for (int i = 0; i < optixLaunchParams.pmCount[hashId]; i++) {
						vec3f dir = optixLaunchParams.pm[start + i].position - point;
						float sq_dist = dot(dir, dir);
						if (sq_dist <= sq_r)
							totalPower += optixLaunchParams.pm[start + i].power;
					}
				}
			}
		}
		return totalPower;
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
		// compute geometry normal and normalize it
		// ------------------------------------------------------------------
		const vec3f& A = sbtData.vertex[index.x];
		const vec3f& B = sbtData.vertex[index.y];
		const vec3f& C = sbtData.vertex[index.z];
		vec3f Ng = cross(B - A, C - A);
		Ng = normalize(Ng);

		// ------------------------------------------------------------------
		// compute ray intersection point
		// ------------------------------------------------------------------
		const vec3f rayOrig = optixGetWorldRayOrigin();
		const vec3f rayDir = optixGetWorldRayDirection();
		const float rayT = optixGetRayTmax();
		vec3f hitPoint = rayOrig + rayDir * rayT;

		// ------------------------------------------------------------------
		// face-forward normals
		// ------------------------------------------------------------------
		vec3f N = Ng;
		float cosi = dot(rayDir, Ng);
		if (cosi > 0.f) Ng = -Ng;

		//if (ix == 600 && iy == 400) {
		//	printf("befor %f %f %f normal %f %f %f hit %f %f %f depth %i\n", N.x, N.y, N.z, Ng.x, Ng.y, Ng.z, hitPoint.x, hitPoint.y, hitPoint.z, prd.depth);
		//}
		
		

		// start with some ambient term
		vec3f pixelColor(0.f);

		// ------------------------------------------------------------------
		// compute multiple diffuse
		// ------------------------------------------------------------------
		if (sbtData.color != vec3f(0)) {
			float radius;
			if (optixLaunchParams.onlyPhotons)
			{
				radius = MAX_RADIUS_ONLY_PHOTONS;
			}
			else
			{
				radius = optixLaunchParams.maxRadius;
			}
			vec3f totalPower = nearest_photons_hash_list(hitPoint, radius);
			vec3f brdf = sbtData.color / M_PI;
			//printf("%f %f %f \n", totalPower.x, totalPower.y, totalPower.z);
			float delta_a = M_PI * radius * radius;
			if (optixLaunchParams.onlyPhotons)
			{
				pixelColor += totalPower / delta_a;
			}
			else
			{
				// sin filtros
				pixelColor += (totalPower * brdf) / delta_a;
			}

			//pixelColor += totalPower;

			// filtro gauss
				//pixelColor += totalPower * brdf;
		}

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
				//float atenuation = 1;
				vec3f base = lightVisibility * atenuation * optixLaunchParams.light.intensity;

				vec3f reflectDir = reflect(-lightDir, Ng);
				float RdotV = fmaxf(0.f, dot(reflectDir, -rayDir));

				if (!optixLaunchParams.onlyPhotons)
				{
					pixelColor += base * sbtData.color * (NdotL / NUM_LIGHT_SAMPLES);
					pixelColor += base * sbtData.specular * pow(RdotV, sbtData.phong) / NUM_LIGHT_SAMPLES;
				}
				//printf("%f %f %f\n", pixelColor.x, pixelColor.y, pixelColor.z);
			}
		}

		///*Regresar si la prof. es excesiva */
		if (prd.depth < optixLaunchParams.maxDepth) {
			prd.depth += 1;

			if (!optixLaunchParams.onlyPhotons)
			{
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

				if (sbtData.transmission != vec3f(0)) {
					PRD refraction;
					refraction.pixelColor = vec3f(0.f);
					refraction.depth = prd.depth;
					refraction.currentIor = prd.currentIor;

					/*float cosi = dot(rayDir, Ng);*/
					float etai = 1, etat = sbtData.ior;
					vec3f n = Ng;
					if (cosi < 0) { 
						cosi = -cosi; 
					} else {
						float tmp = etai;
						etai = etat;
						etat = tmp;
						//n = -Ng; already done before
						//printf("inside");
					}

					float eta = etai / etat;
					float k = 1 - eta * eta * (1 - cosi * cosi);
					vec3f refrDir = (k < 0) ? vec3f(0) : eta * rayDir + (eta * cosi - sqrtf(k)) * n;

					uint32_t u0, u1;
					packPointer(&refraction, u0, u1);

					optixTrace(optixLaunchParams.traversable,
						hitPoint,
						refrDir,
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

			//// refraction component
			//if (sbtData.transmission != vec3f(0)) {
			//	float nit;
			//	PRD refraction;
			//	refraction.pixelColor = vec3f(0.f);
			//	refraction.depth = prd.depth;
			//	refraction.currentIor = prd.currentIor;

			//	// ray comes from outside surface
			//	float cosi = dot(rayDir, Ng);
			//	if (cosi < 0) {
			//		nit = 1 / sbtData.ior;
			//		//nit = prd.currentIor / sbtData.ior;
			//		cosi = -cosi;
			//		refraction.currentIor = sbtData.ior;
			//		//printf("holas");
			//	}
			//	// ray comes from inside surface
			//	else {
			//		// we assume that when it leaves the surface, it goes into the air
			//		nit = sbtData.ior;
			//		Ng = -Ng;
			//		//refraction.currentIor = 1;
			//	}

			//	float testrit = nit * nit * (1 - cosi * cosi);

			//	if (testrit <= 1) {
			//		vec3f refractionDir = nit * rayDir + (nit * cosi - sqrtf(1 - testrit)) * Ng;

			//		uint32_t u0, u1;
			//		packPointer(&refraction, u0, u1);

			//		optixTrace(optixLaunchParams.traversable,
			//			hitPoint,
			//			refractionDir,
			//			1e-4f,  // tmin
			//			1e20f,  // tmax
			//			0.0f,   // rayTime
			//			OptixVisibilityMask(255),
			//			OPTIX_RAY_FLAG_DISABLE_ANYHIT,//OPTIX_RAY_FLAG_NONE,
			//			RADIANCE_RAY_TYPE,            // SBT offset
			//			RAY_TYPE_COUNT,               // SBT stride
			//			RADIANCE_RAY_TYPE,            // missSBTIndex 
			//			u0, u1
			//		);
			//		pixelColor += sbtData.transmission * refraction.pixelColor;
			//	}
			//}
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

		//printf("int %i\n", optixLaunchParams.solo[0]);
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
