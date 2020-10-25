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
#include "ONB.h"

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

	extern "C" __global__ void __closesthit__caustic()
	{
	}

	extern "C" __global__ void __anyhit__caustic()
	{
	}

	extern "C" __global__ void __miss__caustic()
	{
	}

	extern "C" __global__ void __raygen__renderCaustic()
	{
	}

}
