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

// our own classes, partly shared between host and device
#include "CUDABuffer.h"
#include "LaunchParams.h"
#include "Model.h"

/*! \namespace osc - Optix Siggraph Course */
namespace osc {

	struct Camera {
		/*! camera position - *from* where we are looking */
		vec3f from;
		/*! which point we are looking *at* */
		vec3f at;
		/*! general up-vector */
		vec3f up;
	};

	/*! a sample OptiX-7 renderer that demonstrates how to set up
		context, module, programs, pipeline, SBT, etc, and perform a
		valid launch that renders some pixel (using a simple test
		pattern, in this case */
	class SampleRenderer
	{
		// ------------------------------------------------------------------
		// publicly accessible interface
		// ------------------------------------------------------------------
	public:
		/*! constructor - performs all setup, including initializing
		  optix, creates module, pipeline, programs, SBT, etc. */
		SampleRenderer(const Model* model, const PointLight& light, std::string objFileName,
			int numPhotonSamples, int maxDepth, float radius, int antialiasingLevel);

		/*! render one frame */
		void render();

		/*! resize frame buffer to given resolution */
		void resize(const vec2i& newSize);

		/*! download the rendered color buffer */
		void downloadPixels(uint32_t h_pixels[]);
		void downloadPhotons(PhotonPrint h_pixels[]);

		/*! set camera to render with */
		void setCamera(const Camera& camera);

		/*! set the params to render with */
		void setParams(int numPhotonSamples, int maxDepth, float radius, int antialiasingLevel);

		/*! perform the first pass of photon mapping */
		void photonPass();
		void causticsPass();

		/*! renders only the photons for the next frame */
		void renderPhotons();

	protected:
		// ------------------------------------------------------------------
		// internal helper functions
		// ------------------------------------------------------------------

		/*! helper function that initializes optix and checks for errors */
		void initOptix();

		/*! creates and configures a optix device context (in this simple
		  example, only for the primary GPU device) */
		void createContext();

		/*! creates the module that contains all the programs we are going
		  to use. in this simple example, we use a single module from a
		  single .cu file, using a single embedded ptx string */
		void createModule();

		/*! does all setup for the raygen program(s) we are going to use */
		void createRaygenPrograms();

		/*! does all setup for the miss program(s) we are going to use */
		void createMissPrograms();

		/*! does all setup for the hitgroup program(s) we are going to use */
		void createHitgroupPrograms();

		/*! assembles the full pipeline of all programs */
		void createPipeline();

		/*! constructs the shader binding table */
		void buildSBTrender();
		void buildSBTglobal();
		void buildSBTcaustics();

		/*! build an acceleration structure for the given triangle mesh */
		OptixTraversableHandle buildAccel();

		/*! upload textures, and create cuda texture objects for them */
		void createTextures();

		std::string getConfigStr();

		bool loadPhotonMap(std::vector<PhotonPrint>& photons, std::vector<int>& counts, std::vector<int>& starts);
		void savePhotonMap(std::vector<PhotonPrint> photons, std::vector<int> counts, std::vector<int> starts);
	protected:
		/*! @{ CUDA device context and stream that optix pipeline will run
			on, as well as device properties for this device */
		CUcontext          cudaContext;
		CUstream           stream;
		cudaDeviceProp     deviceProps;
		/*! @} */

		//! the optix context that our pipeline will run in.
		OptixDeviceContext optixContext;

		/*! @{ the pipeline we're building */
		OptixPipeline               pipeline;
		OptixPipelineCompileOptions pipelineCompileOptions;
		OptixPipelineLinkOptions    pipelineLinkOptions;
		/*! @} */

		/*! @{ the module that contains out device programs */
		OptixModule                 renderModule;
		OptixModule                 globalModule;
		OptixModule                 causticsModule;
		OptixModuleCompileOptions   moduleCompileOptions;
		/* @} */

		/*! vector of all our program(group)s, and the SBT built around
			them */
		std::vector<OptixProgramGroup> raygenPGs;
		CUDABuffer raygenRecordsBufferRender;
		CUDABuffer raygenRecordsBufferGlobal;
		CUDABuffer raygenRecordsBufferCaustics;
		std::vector<OptixProgramGroup> missPGs;
		CUDABuffer missRecordsBufferRender;
		CUDABuffer missRecordsBufferGlobal;
		CUDABuffer missRecordsBufferCaustics;
		std::vector<OptixProgramGroup> hitgroupPGs;
		CUDABuffer hitgroupRecordsBufferRender;
		CUDABuffer hitgroupRecordsBufferGlobal;
		CUDABuffer hitgroupRecordsBufferCaustics;

		OptixShaderBindingTable sbtRender = {};
		OptixShaderBindingTable sbtGlobal = {};
		OptixShaderBindingTable sbtCaustics = {};

		/*! @{ our launch parameters, on the host, and the buffer to store
			them on the device */
		LaunchParams launchParams;
		CUDABuffer   launchParamsBufferRender;
		CUDABuffer   launchParamsBufferGlobal;
		CUDABuffer   launchParamsBufferCaustics;
		/*! @} */

		CUDABuffer colorBuffer;

		/*! the camera we are to render with. */
		Camera lastSetCamera;

		/*! the model we are going to trace rays against */
		const Model* model;

		/*! @{ one buffer per input mesh */
		std::vector<CUDABuffer> vertexBuffer;
		std::vector<CUDABuffer> normalBuffer;
		std::vector<CUDABuffer> texcoordBuffer;
		std::vector<CUDABuffer> indexBuffer;
		/*! @} */

		//! buffer that keeps the (final, compacted) accel structure
		CUDABuffer asBuffer;

		CUDABuffer haltonNumbers;

		/*! @{ one texture object and pixel array per used texture */
		std::vector<cudaArray_t>         textureArrays;
		std::vector<cudaTextureObject_t> textureObjects;
		/*! @} */

		CUDABuffer prePhotonMap;
		CUDABuffer photonMap;

		CUDABuffer preCausticMap;
		CUDABuffer causticMap;

		CUDABuffer projectionMap;

		CUDABuffer pm;
		CUDABuffer pmCount;
		CUDABuffer pmStarts;
	};

} // ::osc
