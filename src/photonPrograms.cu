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
		float currentIor;
	};

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
		PhotonPRD& prd = *getPRD<PhotonPRD>();
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
		Ng = normalize(Ng);

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

		int rand_index = prd.random() * optixLaunchParams.numPhotonSamples;
		//int rand_index = (ix * prd.depth + 1) % NUM_PHOTON_SAMPLES;
		float coin = optixLaunchParams.halton[rand_index].x;
		//float coin = optixLaunchParams.halton[ix].x;

		// diffuse component is color for now
		float Pd = max(sbtData.color * prd.power) / max(prd.power);
		float Ps = max(sbtData.specular * prd.power) / max(prd.power);
		float Pt = max(sbtData.transmission * prd.power) / max(prd.power);
		//printf("color %f %f %f maximo %f prob difuso %f mult %f %f %f\n", sbtData.color.x, sbtData.color.y, sbtData.color.z, max(sbtData.color), Pd, (sbtData.color * prd.power).x, (sbtData.color * prd.power).y, (sbtData.color * prd.power).z);

		prd.depth += 1;
		//if (coin >= 1) printf("coin %f", coin);
		coin = prd.random();
		if (coin <= Pd) {
			//diffuse

			// avoid first diffuse hit
			PhotonPrint pp;
			if (prd.depth > 1) {
				pp.position = hitPoint;
				pp.power = prd.power;
				optixLaunchParams.prePhotonMap[ix * optixLaunchParams.maxDepth + prd.depth - 2] = pp;

				//int hashId = hash(hitPoint, optixLaunchParams.gridSize, optixLaunchParams.lowerBound);
				//optixLaunchParams.pm[hashId] = pp;
				//atomicAdd(&optixLaunchParams.pmCount[hashId], 1);
			}

			if (prd.depth <= optixLaunchParams.maxDepth) {

				PhotonPRD diffuse;
				diffuse.random = prd.random;
				diffuse.depth = prd.depth;
				diffuse.currentIor = prd.currentIor;

				// obtain random direction
				vec3f U, V, W, direction;
				create_onb(Ng, U, V, W);

				sampleUnitHemisphere(optixLaunchParams.halton[rand_index], U, V, W, direction);

				diffuse.power = (prd.power * sbtData.color) / Pd;
				prd.power = (prd.power * sbtData.color) / Pd;
				printf("%f %f %f otro %f %f %f\n", diffuse.power.x, diffuse.power.y, diffuse.power.z, prd.power.x, prd.power.y, prd.power.z);

				uint32_t u0, u1;
				//packPointer(&prd, u0, u1);
				packPointer(&diffuse, u0, u1);

				optixTrace(
					optixLaunchParams.traversable,
					hitPoint,
					direction,
					1.e-4f,							// tmin
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

				//printf("hit %f %f %f dir %f %f %f normal %f %f %f depth %i\n",
				//	hitPoint.x, hitPoint.y, hitPoint.z,
				//	direction.x, direction.y, direction.z,
				//	Ng.x, Ng.y, Ng.z, prd.depth
				//);
			}
		}
		else if (coin <= Pd + Ps) {
			// specular

			if (prd.depth <= optixLaunchParams.maxDepth) {
				PhotonPRD reflection;
				reflection.random = prd.random;
				reflection.depth = prd.depth;
				reflection.currentIor = prd.currentIor;
				uint32_t u0, u1;
				//packPointer(&prd, u0, u1);
				packPointer(&reflection, u0, u1);

				// obtain reflection direction
				vec3f reflectDir = reflect(rayDir, Ng); //rayo en la direccion de reflexion desde punto;

				//prd.power = (prd.power * sbtData.specular) / Ps;
				reflection.power = (prd.power * sbtData.specular) / Ps;

				optixTrace(
					optixLaunchParams.traversable,
					hitPoint,
					reflectDir,
					1.e-4f,							// tmin
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

			if (prd.depth <= optixLaunchParams.maxDepth) {
				PhotonPRD refraction;
				refraction.random = prd.random;
				refraction.depth = prd.depth;
				refraction.currentIor = prd.currentIor;

				uint32_t u0, u1;
				packPointer(&refraction, u0, u1);
				//packPointer(&prd, u0, u1);

				/*float cosi = dot(rayDir, Ng);*/
				float etai = 1, etat = sbtData.ior;
				vec3f n = Ng;
				if (cosi < 0) {
					cosi = -cosi;
				}
				else {
					float tmp = etai;
					etai = etat;
					etat = tmp;
					//n = -Ng; already done before
					//printf("inside");
				}

				float eta = etai / etat;
				float k = 1 - eta * eta * (1 - cosi * cosi);
				if (k >= 0) {
					vec3f refrDir = eta * rayDir + (eta * cosi - sqrtf(k)) * n;

					refraction.power = (prd.power * sbtData.transmission) / Pt;
					//prd.power = (prd.power * sbtData.transmission) / Pt;
				
					optixTrace(
					optixLaunchParams.traversable,
					hitPoint,
					refrDir,
					1e-4f,							// tmin
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
		}
		else {
			// absorption check if need to store diffuse photons
			if (prd.depth > 1) {
				PhotonPrint pp;
				pp.position = hitPoint;
				pp.power = prd.power;
				optixLaunchParams.prePhotonMap[ix * optixLaunchParams.maxDepth + prd.depth - 2] = pp;

				//int hashId = hash(hitPoint, optixLaunchParams.gridSize, optixLaunchParams.lowerBound);
				//optixLaunchParams.pm[hashId] = pp;
				//atomicAdd(&optixLaunchParams.pmCount[hashId], 1);
			}
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
	}

	//------------------------------------------------------------------------------
	// ray gen program - the actual rendering happens in here
	//------------------------------------------------------------------------------
	extern "C" __global__ void __raygen__renderPhoton()
	{
		// compute a test pattern based on pixel ID
		const int ix = optixGetLaunchIndex().x;

		PhotonPRD prd;
		prd.random.init(ix, ix * optixLaunchParams.frame.size.x);
		prd.depth = 0;
		prd.power = optixLaunchParams.light.photonPower;
		prd.currentIor = 1.f;

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
			1e-4f,							// tmin
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

} // ::osc
