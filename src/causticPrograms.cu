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

	struct CausticPRD {
		Random random;
		vec3f power;
		unsigned int depth;
		float currentIor;
		bool hitSpecular;
	};

	extern "C" __global__ void __closesthit__caustic()
	{
		CausticPRD& prd = *getPRD<CausticPRD>();
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
		if (dot(rayDir, Ng) > 0.f) Ng = -Ng;
		Ng = normalize(Ng);

		int rand_index = prd.random() * NUM_PHOTON_SAMPLES;
		float coin = optixLaunchParams.halton[rand_index].x;

		// diffuse component is color for now
		float Pd = max(sbtData.color * prd.power) / max(prd.power);
		float Ps = max(sbtData.specular * prd.power) / max(prd.power);
		float Pt = max(sbtData.transmission * prd.power) / max(prd.power);
		
		prd.depth += 1;
		coin = prd.random();
		if (coin <= Pd) {
			//diffuse
			PhotonPrint pp;
			if (prd.hitSpecular) {
				pp.position = hitPoint;
				pp.power = prd.power;
				optixLaunchParams.preCausticMap[ix + iy * NC] = pp;
				printf("chau\n");
			}
		}
		else if (coin <= Pd + Ps) {
			// specular
			if (prd.depth <= MAX_DEPTH) {
				uint32_t u0, u1;
				packPointer(&prd, u0, u1);

				// obtain reflection direction
				vec3f reflectDir = reflect(rayDir, Ng); //rayo en la direccion de reflexion desde punto;

				prd.power = (prd.power * sbtData.specular) / Ps;
				prd.hitSpecular = true;
				printf("hola\n");

				optixTrace(
					optixLaunchParams.traversable,
					hitPoint,
					reflectDir,
					1.e-4f, // tmin
					1e20f,							// tmax
					0.0f,							// rayTime
					OptixVisibilityMask(255),
					OPTIX_RAY_FLAG_DISABLE_ANYHIT,	// OPTIX_RAY_FLAG_NONE,
					CAUSTIC_RAY_TYPE,				// SBT offset
					RAY_TYPE_COUNT,					// SBT stride
					CAUSTIC_RAY_TYPE,				// missSBTIndex
					u0, u1
				);
			}
		}
		else if (coin <= Pd + Ps + Pt) {
			// transmission
			if (prd.depth <= MAX_DEPTH) {
				uint32_t u0, u1;
				packPointer(&prd, u0, u1);
				printf("hola2\n");
				float nit;
				float cosi = dot(rayDir, Ng);

				if (cosi < 0) { // ray comes from outside surface
					nit = prd.currentIor / sbtData.ior;
					cosi = -cosi;
					prd.currentIor = sbtData.ior;
				}
				else { // ray comes from inside surface
					nit = sbtData.ior;
				}

				float testrit = nit * nit * (1 - cosi * cosi);
				if (testrit <= 1) {
					// obtain refraction direction
					vec3f refractionDir = nit * rayDir + (nit * cosi - sqrtf(1 - testrit)) * Ng;

					prd.power = (prd.power * sbtData.transmission) / Pt;
					prd.hitSpecular = true;

					optixTrace(
						optixLaunchParams.traversable,
						hitPoint,
						refractionDir,
						1e-4f,							// tmin
						1e20f,							// tmax
						0.0f,							// rayTime
						OptixVisibilityMask(255),
						OPTIX_RAY_FLAG_DISABLE_ANYHIT,	// OPTIX_RAY_FLAG_NONE,
						CAUSTIC_RAY_TYPE,				// SBT offset
						RAY_TYPE_COUNT,					// SBT stride
						CAUSTIC_RAY_TYPE,				// missSBTIndex
						u0, u1
					);
				}
			}
		}
		else {
			// absorption check if need to store diffuse photons
			PhotonPrint pp;
			if (prd.hitSpecular) {
				pp.position = hitPoint;
				pp.power = prd.power;
				optixLaunchParams.preCausticMap[ix + iy * NC] = pp;
				printf("chau\n");
			}
		}
	}

	extern "C" __global__ void __anyhit__caustic()
	{
	}

	extern "C" __global__ void __miss__caustic()
	{
		printf("miss\n");
	}

	extern "C" __global__ void __raygen__renderCaustic()
	{
		const int ix = optixGetLaunchIndex().x;
		const int iy = optixGetLaunchIndex().y;

		CausticPRD prd;
		prd.random.init(ix, ix * optixLaunchParams.frame.size.x);
		prd.depth = 0;
		prd.power = optixLaunchParams.light.photonPower;
		prd.currentIor = 1;


		uint32_t u0, u1;
		packPointer(&prd, u0, u1);
		
		//Ncell = 10000 % cantidad de fotones lanzados por celda
		if (optixLaunchParams.projectionMap[ix + NC * iy] == 1) { // hit a specular
			float xmin = -1 + 2 * (ix - 1) / NC;
			float xmax = xmin + 2 / NC;
			float ymin = -1 + 2 * (iy - 1) / NC;
			float ymax = ymin + 2 / NC;

			float xmin2 = xmin * xmin;
			float xmax2 = xmax * xmax;
			float ymin2 = ymin * ymin;
			float ymax2 = ymax * ymax;
				printf("entra\n");
			if ((xmin2 + ymin2) < 1 || (xmin2 + ymax2) < 1 || (xmax2 + ymin2) < 1 || (xmax2 + ymax2) < 1) {
				//% si alguna de las 4 esquinas de la celda está dentro del círculo de radio 1
				for (int k = 0; k < NUM_CAUSTIC_PER_CELL; k++) {
					float x = xmin + 2 * prd.random() / NC; 
					float y = ymin + 2 * prd.random() / NC;
					int z2 = 1 - x * x - y * y;
					z2 = z2 > 0.0f ? sqrtf(z2) : 0.0f;
					vec3f direction = optixLaunchParams.light.normal * vec3f(x, y, z2);

					prd.power = optixLaunchParams.light.photonPower;
					prd.depth = 0;
					prd.currentIor = 1;

					optixTrace(
						optixLaunchParams.traversable,
						optixLaunchParams.light.origin,
						direction,
						1e-4f,							// tmin
						1e20f,							// tmax
						0.0f,							// rayTime
						OptixVisibilityMask(255),
						OPTIX_RAY_FLAG_DISABLE_ANYHIT,	// OPTIX_RAY_FLAG_NONE,
						CAUSTIC_RAY_TYPE,				// SBT offset
						RAY_TYPE_COUNT,					// SBT stride
						CAUSTIC_RAY_TYPE,				// missSBTIndex
						//prd.depth						// reinterpret_cast<unsigned int&>(prd.depth)
						u0, u1
					);
				}
			}
		}
	}
}
