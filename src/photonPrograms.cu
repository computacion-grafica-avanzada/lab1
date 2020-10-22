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

	struct PhotonPRD {
		Random random;
		vec3f power;
		unsigned int depth;
	};

	//static __device__ void insert_if_near(PhotonPrint*& res, int max_p, PhotonPrint candidate) {
	//	for (int i = 0; i < max_p; i++) {
	//		if 
	//	}
	//}

	//------------------------------------------------------------------------------
	// closest hit and anyhit programs for radiance-type rays.
	//
	// Note eventually we will have to create one pair of those for each
	// ray type and each geometry type we want to render; but this
	// simple example doesn't use any actual geometries yet, so we only
	// create a single, dummy, set of them (we do have to have at least
	// one group of them to set up the SBT)
	//------------------------------------------------------------------------------

	extern "C" __global__ void __closesthit__photon()
	{
		PhotonPRD& prd = *(PhotonPRD*)getPRD<PhotonPRD>();
		const TriangleMeshSBTData& sbtData = *(const TriangleMeshSBTData*)optixGetSbtDataPointer();
		const int ix = optixGetLaunchIndex().x;

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


		int rand_index = (ix * prd.depth) % NUM_PHOTON_SAMPLES;
		float coin = optixLaunchParams.halton[rand_index].x;

		// diffuse component is color for now
		float Pd = max(sbtData.color * prd.power) / max(prd.power);
		float Ps = max(sbtData.specular * prd.power) / max(prd.power);
		float Pt = max(sbtData.transmission * prd.power) / max(prd.power);
		//printf("color %f %f %f maximo %f prob difuso %f mult %f %f %f\n", sbtData.color.x, sbtData.color.y, sbtData.color.z, max(sbtData.color), Pd, (sbtData.color * prd.power).x, (sbtData.color * prd.power).y, (sbtData.color * prd.power).z);

		prd.depth += 1;
		if (coin <= Pd) {
			//diffuse

			// avoid first diffuse hit
			if (prd.depth > 1) {
				PhotonPrint pp;
				pp.position = hitPoint;
				pp.direction = rayDir;
				pp.power = prd.power;
				optixLaunchParams.prePhotonMap[ix * MAX_DEPTH + prd.depth - 2] = pp;
			}

			if (prd.depth <= MAX_DEPTH) {
				uint32_t u0, u1;
				packPointer(&prd, u0, u1);

				// obtain random direction
				vec3f U, V, W, direction;
				create_onb(Ng, U, V, W);

				sampleUnitHemisphere(optixLaunchParams.halton[rand_index].y, U, V, W, direction);

				//printf("direction %f %f %f \n", direction.x, direction.y, direction.z);

				prd.power = (prd.power * sbtData.color) / Pd;

				optixTrace(
					optixLaunchParams.traversable,
					hitPoint,
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
		else if (coin <= Pd + Ps) {
			// specular

			if (prd.depth <= MAX_DEPTH) {
				uint32_t u0, u1;
				packPointer(&prd, u0, u1);

				// obtain reflection direction
				vec3f reflectDir = reflect(-rayDir, Ng); //rayo en la direccion de reflexion desde punto;

				prd.power = (prd.power * sbtData.specular) / Ps;

				optixTrace(
					optixLaunchParams.traversable,
					hitPoint,
					reflectDir,
					0.f,							// tmin
					1e20f,							// tmax
					0.0f,							// rayTime
					OptixVisibilityMask(255),
					OPTIX_RAY_FLAG_DISABLE_ANYHIT,	// OPTIX_RAY_FLAG_NONE,
					PHOTON_RAY_TYPE,				// SBT offset
					RAY_TYPE_COUNT,					// SBT stride
					PHOTON_RAY_TYPE,				// missSBTIndex 
					u0, u1
				);
			}
		}
		else if (coin <= Pd + Ps + Pt) {
			// transmission

			//if (prd.depth <= MAX_DEPTH) {
			//	uint32_t u0, u1;
			//	packPointer(&prd, u0, u1);

			//	// obtain reflection direction
			//	vec3f reflectDir = reflect(-rayDir, Ng); //rayo en la direccion de reflexion desde punto;

			//	prd.power = (prd.power * sbtData.specular) / Ps;

			//	optixTrace(
			//		optixLaunchParams.traversable,
			//		hitPoint,
			//		reflectDir,
			//		0.f,							// tmin
			//		1e20f,							// tmax
			//		0.0f,							// rayTime
			//		OptixVisibilityMask(255),
			//		OPTIX_RAY_FLAG_DISABLE_ANYHIT,	// OPTIX_RAY_FLAG_NONE,
			//		PHOTON_RAY_TYPE,				// SBT offset
			//		RAY_TYPE_COUNT,					// SBT stride
			//		PHOTON_RAY_TYPE,				// missSBTIndex 
			//		u0, u1
			//	);
			//}
		}
		else {
			// absorption check if need to store diffuse photons
		}


		// ruleta rusa
		// en el caso difuso si depth es 0 no se guarda marca
		// chequear depth+1 menor a MAX
		// tirar rayos acorde a reflexion difusa, specular, transmision o nada si se absorbe

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

	extern "C" __global__ void __miss__photon()
	{
		PhotonPRD& prd = *(PhotonPRD*)getPRD<PhotonPRD>();
		//prd.depth = prd.random() * 10;

		//without random
		//PhotonPRD prd = getPhotonPRD();
		//prd.depth = 100;
		//setPhotonPRD(prd);
		//printf("miss %i \n", prd.depth);
	}

	//------------------------------------------------------------------------------
	// ray gen program - the actual rendering happens in here
	//------------------------------------------------------------------------------
	extern "C" __global__ void __raygen__renderPhoton()
	{
		// compute a test pattern based on pixel ID
		const int ix = optixGetLaunchIndex().x;

		Random ran;
		ran.init(ix, ix * optixLaunchParams.frame.size.x);

		//printf("\n random 1: %f 2: %f 3: %f ix: %d hax: %f hay: %f\n", ran(), ran(), ran(), ix, optixLaunchParams.halton[ix].x, optixLaunchParams.halton[ix].y);

		PhotonPRD prd;
		prd.random.init(ix, ix * optixLaunchParams.frame.size.x);
		prd.depth = 0;
		prd.power = optixLaunchParams.light.photonPower;
		//printf("power photon %f %f %f \n", prd.power.x, prd.power.y, prd.power.z);

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
			u0, u1
		);

		//printf("profundidad %i\n", prd.depth);
	}

} // ::osc
