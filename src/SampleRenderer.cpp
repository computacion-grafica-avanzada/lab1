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

#include "SampleRenderer.h"
#include "LaunchParams.h"
// this include may only appear in a single source file:
#include <optix_function_table_definition.h>
#include "halton.h"
#include "halton_seq.h"

#include <algorithm>
#include <iostream>
#include <iterator>
#include <vector>
#include <fstream> // looks like we need this too
#include <chrono>

/*! \namespace osc - Optix Siggraph Course */
namespace osc
{

	extern "C" char embedded_ptx_code[];
	extern "C" char photon_ptx_code[];
	extern "C" char caustic_ptx_code[];

	/*! SBT record for a raygen program */
	struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) RaygenRecord
	{
		__align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
		// just a dummy value - later examples will use more interesting
		// data here
		void* data;
	};

	/*! SBT record for a miss program */
	struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) MissRecord
	{
		__align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
		// just a dummy value - later examples will use more interesting
		// data here
		void* data;
	};

	/*! SBT record for a hitgroup program */
	struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) HitgroupRecord
	{
		__align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
		TriangleMeshSBTData data;
	};

	/*! constructor - performs all setup, including initializing
		  optix, creates module, pipeline, programs, SBT, etc. */
	SampleRenderer::SampleRenderer(const Model* model, const PointLight& light, std::string objFileName,
		int numPhotonSamples, int maxDepth, float radius, int antialiasingLevel)
		: model(model)
	{
		initOptix();

		launchParams.objFileName = objFileName.c_str();
		launchParams.numPhotonSamples = numPhotonSamples;
		launchParams.maxDepth = maxDepth;
		launchParams.maxRadius = radius;
		launchParams.antialiasingLevel = antialiasingLevel;

		std::cout << "#osc: setting hash grid ..." << std::endl;
		vec3f cells = model->bounds.size() / vec3f(launchParams.maxRadius);
		vec3i gridSize((int)ceilf(cells.x), (int)ceilf(cells.y), (int)ceilf(cells.z));

		launchParams.gridSize = gridSize;
		launchParams.lowerBound = model->bounds.lower;

		std::cout << "#osc: creating light ..." << std::endl;
		launchParams.light.origin = light.origin;
		launchParams.light.normal = light.normal;
		launchParams.light.intensity = light.power;
		launchParams.light.photonPower = light.power / light.numberPhotons;
		//launchParams.light.photonPower *= 10;

		std::cout << launchParams.light.photonPower << std::endl;

		std::cout << "#osc: creating optix context ..." << std::endl;
		createContext();

		std::cout << "#osc: setting up module ..." << std::endl;
		createModule();

		std::cout << "#osc: creating raygen programs ..." << std::endl;
		createRaygenPrograms();
		std::cout << "#osc: creating miss programs ..." << std::endl;
		createMissPrograms();
		std::cout << "#osc: creating hitgroup programs ..." << std::endl;
		createHitgroupPrograms();

		launchParams.traversable = buildAccel();

		std::cout << "#osc: setting up optix pipeline ..." << std::endl;
		createPipeline();

		createTextures();

		std::cout << "#osc: building SBT ..." << std::endl;
		buildSBTrender();
		buildSBTglobal();
		buildSBTcaustics();

		launchParamsBufferRender.alloc(sizeof(launchParams));
		std::cout << "#osc: context, module, pipeline, etc, all set up ..." << std::endl;

		std::cout << GDT_TERMINAL_GREEN;
		std::cout << "#osc: Optix 7 Sample fully set up" << std::endl;
		std::cout << GDT_TERMINAL_DEFAULT;

		launchParams.onlyPhotons = true;


		auto start = std::chrono::high_resolution_clock::now();
		std::cout << "#osc: starting photon pass" << std::endl;
		photonPass();
		std::cout << "#osc: finished photon pass" << std::endl;
		auto finish = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> elapsed = finish - start;
		std::cout << "Photon pass elapsed time: " << elapsed.count() << " s\n";
		// here would be the caustics pass
		//std::cout << "#osc: starting caustic pass" << std::endl;
		//causticsPass();
		//std::cout << "#osc: finished caustic pass" << std::endl;
	}

	int max(int a, int b)
	{
		return (a > b) ? a : b;
	}

	//void SampleRenderer::causticsPass()
	//{
	//	// resize our cuda frame buffer
	//	preCausticMap.resize(NUM_CAUSTIC_PER_CELL * NC * NC * sizeof(PhotonPrint));
	//	launchParams.preCausticMap = (PhotonPrint*)preCausticMap.d_pointer();

	//	// set launchParams for launch
	//	launchParamsBufferCaustics.alloc(sizeof(launchParams));
	//	launchParamsBufferCaustics.upload(&launchParams, 1);

	//	OPTIX_CHECK(optixLaunch(
	//		pipeline,
	//		stream,
	//		launchParamsBufferCaustics.d_pointer(),
	//		launchParamsBufferCaustics.sizeInBytes,
	//		&sbtCaustics,
	//		NC,
	//		NC,
	//		1));

	//	std::vector<PhotonPrint> photonsData(NUM_CAUSTIC_PER_CELL * NC * NC);
	//	preCausticMap.download(photonsData.data(), photonsData.size());
	//}

	void SampleRenderer::photonPass()
	{
		int totalCells = launchParams.gridSize.x * launchParams.gridSize.y * launchParams.gridSize.z;
		std::vector<int> pmGridCount(totalCells, 0);	// amount of photons per cell
		std::vector<int> pmGridStarts(totalCells, 0);
		std::vector<PhotonPrint> traces;
		if (loadPhotonMap(traces, pmGridCount, pmGridStarts)) {
			std::cout << "Loading photon map" << '\n';
			// upload buffers to GPU
			pmStarts.alloc_and_upload(pmGridStarts);
			launchParams.pmStarts = (int*)pmStarts.d_pointer();

			pmCount.alloc_and_upload(pmGridCount);
			launchParams.pmCount = (int*)pmCount.d_pointer();

			pm.alloc_and_upload(traces);
			launchParams.pm = (PhotonPrint*)pm.d_pointer();
		}
		else {
			std::cout << "#osc: creating halton numbers ..." << std::endl;
			std::vector<vec2f> haltons;
			for (int i = 0; i < launchParams.numPhotonSamples; i++)
			{
				if (launchParams.numPhotonSamples > 1000000)
				{
					double* numeros = halton(i + 181472, 2);
					haltons.push_back(vec2f((float)numeros[0], (float)numeros[1]));
				}
				else
				{
					haltons.push_back(vec2f(HALTON_1[i], HALTON_2[i]));
				}
			}
			haltonNumbers.alloc_and_upload(haltons);
			launchParams.halton = (vec2f*)haltonNumbers.d_pointer();

			// resize our cuda frame buffer
			prePhotonMap.resize(launchParams.numPhotonSamples * launchParams.maxDepth * sizeof(PhotonPrint));
			launchParams.prePhotonMap = (PhotonPrint*)prePhotonMap.d_pointer();

			// set launchParams for launch
			launchParamsBufferGlobal.alloc(sizeof(launchParams));
			launchParamsBufferGlobal.upload(&launchParams, 1);

			OPTIX_CHECK(optixLaunch(
				pipeline,
				stream,
				launchParamsBufferGlobal.d_pointer(),
				launchParamsBufferGlobal.sizeInBytes,
				&sbtGlobal,
				launchParams.numPhotonSamples,
				1,
				1));

			// obtain photon traces
			std::vector<PhotonPrint> photonsData(launchParams.numPhotonSamples * launchParams.maxDepth);
			prePhotonMap.download(photonsData.data(), photonsData.size());
			prePhotonMap.free();
			haltonNumbers.free();

			// filter empty slots and populate grid
			PhotonPrint nu = { vec3f(0), vec3f(0) };

			std::vector<std::vector<PhotonPrint>> multiple(totalCells); // list of photons per cell
			for (auto ph : photonsData)
			{
				if (ph != nu)
				{
					vec3f local = ph.position - launchParams.lowerBound;
					vec3i G(
						max(0, (int)(local.x / launchParams.maxRadius)),
						max(0, (int)(local.y / launchParams.maxRadius)),
						max(0, (int)(local.z / launchParams.maxRadius)));
					int hashId = G.x + G.y * launchParams.gridSize.x + G.z * launchParams.gridSize.x * launchParams.gridSize.y;
					multiple[hashId].push_back(ph);
					pmGridCount[hashId] += 1;
				}
			}
			photonsData.clear();

			// linearilize photon map
			for (int i = 0; i < multiple.size(); i++)
			{
				pmGridStarts[i] = traces.size();
				for (auto trace : multiple[i])
				{
					traces.push_back(trace);
				}
			}

			// upload buffers to GPU
			pmStarts.alloc_and_upload(pmGridStarts);
			launchParams.pmStarts = (int*)pmStarts.d_pointer();

			pmCount.alloc_and_upload(pmGridCount);
			launchParams.pmCount = (int*)pmCount.d_pointer();

			pm.alloc_and_upload(traces);
			launchParams.pm = (PhotonPrint*)pm.d_pointer();

			savePhotonMap(traces, pmGridCount, pmGridStarts);
		}
	}

	void SampleRenderer::createTextures()
	{
		int numTextures = (int)model->textures.size();

		textureArrays.resize(numTextures);
		textureObjects.resize(numTextures);

		for (int textureID = 0; textureID < numTextures; textureID++)
		{
			auto texture = model->textures[textureID];

			cudaResourceDesc res_desc = {};

			cudaChannelFormatDesc channel_desc;
			int32_t width = texture->resolution.x;
			int32_t height = texture->resolution.y;
			int32_t numComponents = 4;
			int32_t pitch = width * numComponents * sizeof(uint8_t);
			channel_desc = cudaCreateChannelDesc<uchar4>();

			cudaArray_t& pixelArray = textureArrays[textureID];
			CUDA_CHECK(MallocArray(&pixelArray,
				&channel_desc,
				width, height));

			CUDA_CHECK(Memcpy2DToArray(pixelArray,
				/* offset */ 0, 0,
				texture->pixel,
				pitch, pitch, height,
				cudaMemcpyHostToDevice));

			res_desc.resType = cudaResourceTypeArray;
			res_desc.res.array.array = pixelArray;

			cudaTextureDesc tex_desc = {};
			tex_desc.addressMode[0] = cudaAddressModeWrap;
			tex_desc.addressMode[1] = cudaAddressModeWrap;
			tex_desc.filterMode = cudaFilterModeLinear;
			tex_desc.readMode = cudaReadModeNormalizedFloat;
			tex_desc.normalizedCoords = 1;
			tex_desc.maxAnisotropy = 1;
			tex_desc.maxMipmapLevelClamp = 99;
			tex_desc.minMipmapLevelClamp = 0;
			tex_desc.mipmapFilterMode = cudaFilterModePoint;
			tex_desc.borderColor[0] = 1.0f;
			tex_desc.sRGB = 0;

			// Create texture object
			cudaTextureObject_t cuda_tex = 0;
			CUDA_CHECK(CreateTextureObject(&cuda_tex, &res_desc, &tex_desc, nullptr));
			textureObjects[textureID] = cuda_tex;
		}
	}

	OptixTraversableHandle SampleRenderer::buildAccel()
	{
		const int numMeshes = (int)model->meshes.size();
		vertexBuffer.resize(numMeshes);
		normalBuffer.resize(numMeshes);
		texcoordBuffer.resize(numMeshes);
		indexBuffer.resize(numMeshes);

		OptixTraversableHandle asHandle{ 0 };

		// ==================================================================
		// triangle inputs
		// ==================================================================
		std::vector<OptixBuildInput> triangleInput(numMeshes);
		std::vector<CUdeviceptr> d_vertices(numMeshes);
		std::vector<CUdeviceptr> d_indices(numMeshes);
		std::vector<uint32_t> triangleInputFlags(numMeshes);

		for (int meshID = 0; meshID < numMeshes; meshID++)
		{
			// upload the model to the device: the builder
			TriangleMesh& mesh = *model->meshes[meshID];
			vertexBuffer[meshID].alloc_and_upload(mesh.vertex);
			indexBuffer[meshID].alloc_and_upload(mesh.index);
			if (!mesh.normal.empty())
				normalBuffer[meshID].alloc_and_upload(mesh.normal);
			if (!mesh.texcoord.empty())
				texcoordBuffer[meshID].alloc_and_upload(mesh.texcoord);

			triangleInput[meshID] = {};
			triangleInput[meshID].type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

			// create local variables, because we need a *pointer* to the
			// device pointers
			d_vertices[meshID] = vertexBuffer[meshID].d_pointer();
			d_indices[meshID] = indexBuffer[meshID].d_pointer();

			triangleInput[meshID].triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
			triangleInput[meshID].triangleArray.vertexStrideInBytes = sizeof(vec3f);
			triangleInput[meshID].triangleArray.numVertices = (int)mesh.vertex.size();
			triangleInput[meshID].triangleArray.vertexBuffers = &d_vertices[meshID];

			triangleInput[meshID].triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
			triangleInput[meshID].triangleArray.indexStrideInBytes = sizeof(vec3i);
			triangleInput[meshID].triangleArray.numIndexTriplets = (int)mesh.index.size();
			triangleInput[meshID].triangleArray.indexBuffer = d_indices[meshID];

			triangleInputFlags[meshID] = 0;

			// in this example we have one SBT entry, and no per-primitive
			// materials:
			triangleInput[meshID].triangleArray.flags = &triangleInputFlags[meshID];
			triangleInput[meshID].triangleArray.numSbtRecords = 1;
			triangleInput[meshID].triangleArray.sbtIndexOffsetBuffer = 0;
			triangleInput[meshID].triangleArray.sbtIndexOffsetSizeInBytes = 0;
			triangleInput[meshID].triangleArray.sbtIndexOffsetStrideInBytes = 0;
		}
		// ==================================================================
		// BLAS setup
		// ==================================================================

		OptixAccelBuildOptions accelOptions = {};
		accelOptions.buildFlags = OPTIX_BUILD_FLAG_NONE | OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
		accelOptions.motionOptions.numKeys = 1;
		accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

		OptixAccelBufferSizes blasBufferSizes;
		OPTIX_CHECK(optixAccelComputeMemoryUsage(optixContext,
			&accelOptions,
			triangleInput.data(),
			(int)numMeshes, // num_build_inputs
			&blasBufferSizes));

		// ==================================================================
		// prepare compaction
		// ==================================================================

		CUDABuffer compactedSizeBuffer;
		compactedSizeBuffer.alloc(sizeof(uint64_t));

		OptixAccelEmitDesc emitDesc;
		emitDesc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
		emitDesc.result = compactedSizeBuffer.d_pointer();

		// ==================================================================
		// execute build (main stage)
		// ==================================================================

		CUDABuffer tempBuffer;
		tempBuffer.alloc(blasBufferSizes.tempSizeInBytes);

		CUDABuffer outputBuffer;
		outputBuffer.alloc(blasBufferSizes.outputSizeInBytes);

		OPTIX_CHECK(optixAccelBuild(optixContext,
			/* stream */ 0,
			&accelOptions,
			triangleInput.data(),
			(int)numMeshes,
			tempBuffer.d_pointer(),
			tempBuffer.sizeInBytes,

			outputBuffer.d_pointer(),
			outputBuffer.sizeInBytes,

			&asHandle,

			&emitDesc, 1));
		CUDA_SYNC_CHECK();

		// ==================================================================
		// perform compaction
		// ==================================================================
		uint64_t compactedSize;
		compactedSizeBuffer.download(&compactedSize, 1);

		asBuffer.alloc(compactedSize);
		OPTIX_CHECK(optixAccelCompact(optixContext,
			/*stream:*/ 0,
			asHandle,
			asBuffer.d_pointer(),
			asBuffer.sizeInBytes,
			&asHandle));
		CUDA_SYNC_CHECK();

		// ==================================================================
		// aaaaaand .... clean up
		// ==================================================================
		outputBuffer.free(); // << the UNcompacted, temporary output buffer
		tempBuffer.free();
		compactedSizeBuffer.free();

		return asHandle;
	}

	/*! helper function that initializes optix and checks for errors */
	void SampleRenderer::initOptix()
	{
		std::cout << "#osc: initializing optix..." << std::endl;

		// -------------------------------------------------------
		// check for available optix7 capable devices
		// -------------------------------------------------------
		cudaFree(0);
		int numDevices;
		cudaGetDeviceCount(&numDevices);
		if (numDevices == 0)
			throw std::runtime_error("#osc: no CUDA capable devices found!");
		std::cout << "#osc: found " << numDevices << " CUDA devices" << std::endl;

		// -------------------------------------------------------
		// initialize optix
		// -------------------------------------------------------
		OPTIX_CHECK(optixInit());
		std::cout << GDT_TERMINAL_GREEN
			<< "#osc: successfully initialized optix... yay!"
			<< GDT_TERMINAL_DEFAULT << std::endl;
	}

	static void context_log_cb(unsigned int level,
		const char* tag,
		const char* message,
		void*)
	{
		fprintf(stderr, "[%2d][%12s]: %s\n", (int)level, tag, message);
	}

	/*! creates and configures a optix device context (in this simple
		  example, only for the primary GPU device) */
	void SampleRenderer::createContext()
	{
		// for this sample, do everything on one device
		const int deviceID = 0;
		CUDA_CHECK(SetDevice(deviceID));
		CUDA_CHECK(StreamCreate(&stream));

		cudaGetDeviceProperties(&deviceProps, deviceID);
		std::cout << "#osc: running on device: " << deviceProps.name << std::endl;

		CUresult cuRes = cuCtxGetCurrent(&cudaContext);
		if (cuRes != CUDA_SUCCESS)
			fprintf(stderr, "Error querying current context: error code %d\n", cuRes);

		OPTIX_CHECK(optixDeviceContextCreate(cudaContext, 0, &optixContext));
		OPTIX_CHECK(optixDeviceContextSetLogCallback(optixContext, context_log_cb, nullptr, 4));
	}

	/*! creates the module that contains all the programs we are going to use. */
	void SampleRenderer::createModule()
	{
		moduleCompileOptions = {};

		pipelineCompileOptions = {};
		pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
		pipelineCompileOptions.usesMotionBlur = false;
		pipelineCompileOptions.numPayloadValues = 2;
		pipelineCompileOptions.numAttributeValues = 2;
		pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
		pipelineCompileOptions.pipelineLaunchParamsVariableName = "optixLaunchParams";

		pipelineLinkOptions.maxTraceDepth = 10;

		const std::string ptxCode = embedded_ptx_code;
		const std::string ptxCodePhoton = photon_ptx_code;
		const std::string ptxCodeCaustic = caustic_ptx_code;

		char log[2048];
		size_t sizeof_log = sizeof(log);
		OPTIX_CHECK(optixModuleCreateFromPTX(
			optixContext,
			&moduleCompileOptions,
			&pipelineCompileOptions,
			ptxCode.c_str(),
			ptxCode.size(),
			log, &sizeof_log,
			&renderModule));

		char log2[2048];
		size_t sizeof_log2 = sizeof(log2);

		OPTIX_CHECK(optixModuleCreateFromPTX(
			optixContext,
			&moduleCompileOptions,
			&pipelineCompileOptions,
			ptxCodePhoton.c_str(),
			ptxCodePhoton.size(),
			log2, &sizeof_log2,
			&globalModule));

		char logCaustics[2048];
		size_t sizeof_logCaustics = sizeof(logCaustics);
		OPTIX_CHECK(optixModuleCreateFromPTX(
			optixContext,
			&moduleCompileOptions,
			&pipelineCompileOptions,
			ptxCodeCaustic.c_str(),
			ptxCodeCaustic.size(),
			logCaustics, &sizeof_logCaustics,
			&causticsModule));
		if (sizeof_log > 1)
			PRINT(log);
	}

	/*! does all setup for the raygen program(s) we are going to use */
	void SampleRenderer::createRaygenPrograms()
	{
		// we do a single ray gen program in this example:
		raygenPGs.resize(RAY_TYPE_COUNT - 1);

		char log[2048];
		size_t sizeof_log = sizeof(log);

		OptixProgramGroupOptions pgOptions = {};
		OptixProgramGroupDesc pgDesc = {};
		pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;

		pgDesc.raygen.module = renderModule;
		pgDesc.raygen.entryFunctionName = "__raygen__renderFrame";
		OPTIX_CHECK(optixProgramGroupCreate(optixContext,
			&pgDesc,
			1,
			&pgOptions,
			log, &sizeof_log,
			&raygenPGs[RADIANCE_RAY_TYPE]));

		pgDesc.raygen.module = globalModule;
		pgDesc.raygen.entryFunctionName = "__raygen__renderPhoton";
		OPTIX_CHECK(optixProgramGroupCreate(optixContext,
			&pgDesc,
			1,
			&pgOptions,
			log, &sizeof_log,
			&raygenPGs[PHOTON_RAY_TYPE]));

		pgDesc.raygen.module = causticsModule;
		pgDesc.raygen.entryFunctionName = "__raygen__renderCaustic";
		OPTIX_CHECK(optixProgramGroupCreate(optixContext,
			&pgDesc,
			1,
			&pgOptions,
			log, &sizeof_log,
			&raygenPGs[CAUSTIC_RAY_TYPE]));

		if (sizeof_log > 1)
			PRINT(log);
	}

	/*! does all setup for the miss program(s) we are going to use */
	void SampleRenderer::createMissPrograms()
	{
		// we do a single ray gen program in this example:
		missPGs.resize(RAY_TYPE_COUNT);

		char log[2048];
		size_t sizeof_log = sizeof(log);

		OptixProgramGroupOptions pgOptions = {};
		OptixProgramGroupDesc pgDesc = {};
		pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;

		// ------------------------------------------------------------------
		// radiance rays
		// ------------------------------------------------------------------
		pgDesc.miss.module = renderModule;
		pgDesc.miss.entryFunctionName = "__miss__radiance";

		OPTIX_CHECK(optixProgramGroupCreate(optixContext,
			&pgDesc,
			1,
			&pgOptions,
			log, &sizeof_log,
			&missPGs[RADIANCE_RAY_TYPE]));
		if (sizeof_log > 1)
			PRINT(log);

		// ------------------------------------------------------------------
		// shadow rays
		// ------------------------------------------------------------------
		pgDesc.miss.module = renderModule;
		pgDesc.miss.entryFunctionName = "__miss__shadow";

		OPTIX_CHECK(optixProgramGroupCreate(optixContext,
			&pgDesc,
			1,
			&pgOptions,
			log, &sizeof_log,
			&missPGs[SHADOW_RAY_TYPE]));
		if (sizeof_log > 1)
			PRINT(log);

		// ------------------------------------------------------------------
		// photon rays
		// ------------------------------------------------------------------
		pgDesc.miss.module = globalModule;
		pgDesc.miss.entryFunctionName = "__miss__photon";

		OPTIX_CHECK(optixProgramGroupCreate(optixContext,
			&pgDesc,
			1,
			&pgOptions,
			log, &sizeof_log,
			&missPGs[PHOTON_RAY_TYPE]));
		if (sizeof_log > 1)
			PRINT(log);

		// ------------------------------------------------------------------
		// caustic rays
		// ------------------------------------------------------------------
		pgDesc.miss.module = causticsModule;
		pgDesc.miss.entryFunctionName = "__miss__caustic";

		OPTIX_CHECK(optixProgramGroupCreate(optixContext,
			&pgDesc,
			1,
			&pgOptions,
			log, &sizeof_log,
			&missPGs[CAUSTIC_RAY_TYPE]));
		if (sizeof_log > 1)
			PRINT(log);
	}

	/*! does all setup for the hitgroup program(s) we are going to use */
	void SampleRenderer::createHitgroupPrograms()
	{
		// for this simple example, we set up a single hit group
		hitgroupPGs.resize(RAY_TYPE_COUNT);

		char log[2048];
		size_t sizeof_log = sizeof(log);

		OptixProgramGroupOptions pgOptions = {};
		OptixProgramGroupDesc pgDesc = {};
		pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;

		// -------------------------------------------------------
		// radiance rays
		// -------------------------------------------------------
		pgDesc.hitgroup.moduleCH = renderModule;
		pgDesc.hitgroup.entryFunctionNameCH = "__closesthit__radiance";

		pgDesc.hitgroup.moduleAH = renderModule;
		pgDesc.hitgroup.entryFunctionNameAH = "__anyhit__radiance";

		OPTIX_CHECK(optixProgramGroupCreate(optixContext,
			&pgDesc,
			1,
			&pgOptions,
			log, &sizeof_log,
			&hitgroupPGs[RADIANCE_RAY_TYPE]));
		if (sizeof_log > 1)
			PRINT(log);

		// -------------------------------------------------------
		// shadow rays
		// -------------------------------------------------------
		pgDesc.hitgroup.moduleCH = renderModule;
		pgDesc.hitgroup.entryFunctionNameCH = "__closesthit__shadow";

		pgDesc.hitgroup.moduleAH = renderModule;
		pgDesc.hitgroup.entryFunctionNameAH = "__anyhit__shadow";

		OPTIX_CHECK(optixProgramGroupCreate(optixContext,
			&pgDesc,
			1,
			&pgOptions,
			log, &sizeof_log,
			&hitgroupPGs[SHADOW_RAY_TYPE]));
		if (sizeof_log > 1)
			PRINT(log);

		// -------------------------------------------------------
		// photon rays
		// -------------------------------------------------------
		pgDesc.hitgroup.moduleCH = globalModule;
		pgDesc.hitgroup.entryFunctionNameCH = "__closesthit__photon";

		pgDesc.hitgroup.moduleAH = globalModule;
		pgDesc.hitgroup.entryFunctionNameAH = "__anyhit__photon";

		OPTIX_CHECK(optixProgramGroupCreate(optixContext,
			&pgDesc,
			1,
			&pgOptions,
			log, &sizeof_log,
			&hitgroupPGs[PHOTON_RAY_TYPE]));
		if (sizeof_log > 1)
			PRINT(log);

		// -------------------------------------------------------
		// caustic rays
		// -------------------------------------------------------
		pgDesc.hitgroup.moduleCH = causticsModule;
		pgDesc.hitgroup.entryFunctionNameCH = "__closesthit__caustic";

		pgDesc.hitgroup.moduleAH = causticsModule;
		pgDesc.hitgroup.entryFunctionNameAH = "__anyhit__caustic";

		OPTIX_CHECK(optixProgramGroupCreate(optixContext,
			&pgDesc,
			1,
			&pgOptions,
			log, &sizeof_log,
			&hitgroupPGs[CAUSTIC_RAY_TYPE]));
		if (sizeof_log > 1)
			PRINT(log);
	}

	/*! assembles the full pipeline of all programs */
	void SampleRenderer::createPipeline()
	{
		std::vector<OptixProgramGroup> programGroups;
		for (auto pg : raygenPGs)
			programGroups.push_back(pg);
		for (auto pg : hitgroupPGs)
			programGroups.push_back(pg);
		for (auto pg : missPGs)
			programGroups.push_back(pg);

		char log[2048];
		size_t sizeof_log = sizeof(log);
		PING;
		PRINT(programGroups.size());
		OPTIX_CHECK(optixPipelineCreate(optixContext,
			&pipelineCompileOptions,
			&pipelineLinkOptions,
			programGroups.data(),
			(int)programGroups.size(),
			log, &sizeof_log,
			&pipeline));
		if (sizeof_log > 1)
			PRINT(log);

		OPTIX_CHECK(optixPipelineSetStackSize(/* [in] The pipeline to configure the stack size for */
			pipeline,
			/* [in] The direct stack size requirement for direct
			callables invoked from IS or AH. */
			2 * 1024,
			/* [in] The direct stack size requirement for direct
			callables invoked from RG, MS, or CH.  */
			2 * 1024,
			/* [in] The continuation stack requirement. */
			2 * 1024,
			/* [in] The maximum depth of a traversable graph
			passed to trace. */
			1));
		if (sizeof_log > 1)
			PRINT(log);
	}

	/*! constructs the shader binding table */
	void SampleRenderer::buildSBTrender()
	{
		// ------------------------------------------------------------------
		// build raygen records
		// ------------------------------------------------------------------
		std::vector<RaygenRecord> raygenRecords;
		//for (int i = 0;i < raygenPGs.size();i++) {
		RaygenRecord rec;
		OPTIX_CHECK(optixSbtRecordPackHeader(raygenPGs[RADIANCE_RAY_TYPE], &rec));
		rec.data = nullptr; /* for now ... */
		raygenRecords.push_back(rec);
		//}
		raygenRecordsBufferRender.alloc_and_upload(raygenRecords);
		sbtRender.raygenRecord = raygenRecordsBufferRender.d_pointer();

		// ------------------------------------------------------------------
		// build miss records
		// ------------------------------------------------------------------
		std::vector<MissRecord> missRecords;
		for (int i = 0; i < missPGs.size(); i++)
		{
			MissRecord rec;
			OPTIX_CHECK(optixSbtRecordPackHeader(missPGs[i], &rec));
			rec.data = nullptr; /* for now ... */
			missRecords.push_back(rec);
		}
		missRecordsBufferRender.alloc_and_upload(missRecords);
		sbtRender.missRecordBase = missRecordsBufferRender.d_pointer();
		sbtRender.missRecordStrideInBytes = sizeof(MissRecord);
		sbtRender.missRecordCount = (int)missRecords.size();

		// ------------------------------------------------------------------
		// build hitgroup records
		// ------------------------------------------------------------------
		int numObjects = (int)model->meshes.size();
		std::vector<HitgroupRecord> hitgroupRecords;
		for (int meshID = 0; meshID < numObjects; meshID++)
		{
			for (int rayID = 0; rayID < RAY_TYPE_COUNT; rayID++)
			{
				auto mesh = model->meshes[meshID];

				HitgroupRecord rec;
				OPTIX_CHECK(optixSbtRecordPackHeader(hitgroupPGs[rayID], &rec));
				rec.data.color = mesh->diffuse;
				rec.data.specular = mesh->specular;
				rec.data.transmission = mesh->transmission;
				rec.data.ior = mesh->ior;
				rec.data.phong = mesh->phong;
				if (mesh->diffuseTextureID >= 0)
				{
					rec.data.hasTexture = true;
					rec.data.texture = textureObjects[mesh->diffuseTextureID];
				}
				else
				{
					rec.data.hasTexture = false;
				}
				rec.data.index = (vec3i*)indexBuffer[meshID].d_pointer();
				rec.data.vertex = (vec3f*)vertexBuffer[meshID].d_pointer();
				rec.data.normal = (vec3f*)normalBuffer[meshID].d_pointer();
				rec.data.texcoord = (vec2f*)texcoordBuffer[meshID].d_pointer();
				hitgroupRecords.push_back(rec);
			}
		}
		hitgroupRecordsBufferRender.alloc_and_upload(hitgroupRecords);
		sbtRender.hitgroupRecordBase = hitgroupRecordsBufferRender.d_pointer();
		sbtRender.hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);
		sbtRender.hitgroupRecordCount = (int)hitgroupRecords.size();
	}

	/*! constructs the shader binding table */
	void SampleRenderer::buildSBTglobal()
	{
		// ------------------------------------------------------------------
		// build raygen records
		// ------------------------------------------------------------------
		std::vector<RaygenRecord> raygenRecords;
		//for (int i = 0;i < raygenPGs.size();i++) {
		RaygenRecord rec;
		OPTIX_CHECK(optixSbtRecordPackHeader(raygenPGs[PHOTON_RAY_TYPE], &rec));
		rec.data = nullptr; /* for now ... */
		raygenRecords.push_back(rec);
		//}
		raygenRecordsBufferGlobal.alloc_and_upload(raygenRecords);
		sbtGlobal.raygenRecord = raygenRecordsBufferGlobal.d_pointer();

		// ------------------------------------------------------------------
		// build miss records
		// ------------------------------------------------------------------
		std::vector<MissRecord> missRecords;
		for (int i = 0; i < missPGs.size(); i++)
		{
			MissRecord rec;
			OPTIX_CHECK(optixSbtRecordPackHeader(missPGs[i], &rec));
			rec.data = nullptr; /* for now ... */
			missRecords.push_back(rec);
		}
		missRecordsBufferGlobal.alloc_and_upload(missRecords);
		sbtGlobal.missRecordBase = missRecordsBufferGlobal.d_pointer();
		sbtGlobal.missRecordStrideInBytes = sizeof(MissRecord);
		sbtGlobal.missRecordCount = (int)missRecords.size();

		// ------------------------------------------------------------------
		// build hitgroup records
		// ------------------------------------------------------------------
		int numObjects = (int)model->meshes.size();
		std::vector<HitgroupRecord> hitgroupRecords;
		for (int meshID = 0; meshID < numObjects; meshID++)
		{
			for (int rayID = 0; rayID < RAY_TYPE_COUNT; rayID++)
			{
				auto mesh = model->meshes[meshID];

				HitgroupRecord rec;
				OPTIX_CHECK(optixSbtRecordPackHeader(hitgroupPGs[rayID], &rec));
				rec.data.color = mesh->diffuse;
				rec.data.specular = mesh->specular;
				rec.data.transmission = mesh->transmission;
				rec.data.ior = mesh->ior;
				rec.data.phong = mesh->phong;
				if (mesh->diffuseTextureID >= 0)
				{
					rec.data.hasTexture = true;
					rec.data.texture = textureObjects[mesh->diffuseTextureID];
				}
				else
				{
					rec.data.hasTexture = false;
				}
				rec.data.index = (vec3i*)indexBuffer[meshID].d_pointer();
				rec.data.vertex = (vec3f*)vertexBuffer[meshID].d_pointer();
				rec.data.normal = (vec3f*)normalBuffer[meshID].d_pointer();
				rec.data.texcoord = (vec2f*)texcoordBuffer[meshID].d_pointer();
				hitgroupRecords.push_back(rec);
			}
		}
		hitgroupRecordsBufferGlobal.alloc_and_upload(hitgroupRecords);
		sbtGlobal.hitgroupRecordBase = hitgroupRecordsBufferGlobal.d_pointer();
		sbtGlobal.hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);
		sbtGlobal.hitgroupRecordCount = (int)hitgroupRecords.size();
	}

	/*! constructs the shader binding table */
	void SampleRenderer::buildSBTcaustics()
	{
		// ------------------------------------------------------------------
		// build raygen records
		// ------------------------------------------------------------------
		std::vector<RaygenRecord> raygenRecords;
		//for (int i = 0;i < raygenPGs.size();i++) {
		RaygenRecord rec;
		OPTIX_CHECK(optixSbtRecordPackHeader(raygenPGs[CAUSTIC_RAY_TYPE], &rec));
		rec.data = nullptr; /* for now ... */
		raygenRecords.push_back(rec);
		//}
		raygenRecordsBufferCaustics.alloc_and_upload(raygenRecords);
		sbtCaustics.raygenRecord = raygenRecordsBufferCaustics.d_pointer();

		// ------------------------------------------------------------------
		// build miss records
		// ------------------------------------------------------------------
		std::vector<MissRecord> missRecords;
		for (int i = 0; i < missPGs.size(); i++)
		{
			MissRecord rec;
			OPTIX_CHECK(optixSbtRecordPackHeader(missPGs[i], &rec));
			rec.data = nullptr; /* for now ... */
			missRecords.push_back(rec);
		}
		missRecordsBufferCaustics.alloc_and_upload(missRecords);
		sbtCaustics.missRecordBase = missRecordsBufferCaustics.d_pointer();
		sbtCaustics.missRecordStrideInBytes = sizeof(MissRecord);
		sbtCaustics.missRecordCount = (int)missRecords.size();

		// ------------------------------------------------------------------
		// build hitgroup records
		// ------------------------------------------------------------------
		int numObjects = (int)model->meshes.size();
		std::vector<HitgroupRecord> hitgroupRecords;
		for (int meshID = 0; meshID < numObjects; meshID++)
		{
			for (int rayID = 0; rayID < RAY_TYPE_COUNT; rayID++)
			{
				auto mesh = model->meshes[meshID];

				HitgroupRecord rec;
				OPTIX_CHECK(optixSbtRecordPackHeader(hitgroupPGs[rayID], &rec));
				rec.data.color = mesh->diffuse;
				rec.data.specular = mesh->specular;
				rec.data.transmission = mesh->transmission;
				rec.data.ior = mesh->ior;
				rec.data.phong = mesh->phong;
				if (mesh->diffuseTextureID >= 0)
				{
					rec.data.hasTexture = true;
					rec.data.texture = textureObjects[mesh->diffuseTextureID];
				}
				else
				{
					rec.data.hasTexture = false;
				}
				rec.data.index = (vec3i*)indexBuffer[meshID].d_pointer();
				rec.data.vertex = (vec3f*)vertexBuffer[meshID].d_pointer();
				rec.data.normal = (vec3f*)normalBuffer[meshID].d_pointer();
				rec.data.texcoord = (vec2f*)texcoordBuffer[meshID].d_pointer();
				hitgroupRecords.push_back(rec);
			}
		}
		hitgroupRecordsBufferCaustics.alloc_and_upload(hitgroupRecords);
		sbtCaustics.hitgroupRecordBase = hitgroupRecordsBufferCaustics.d_pointer();
		sbtCaustics.hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);
		sbtCaustics.hitgroupRecordCount = (int)hitgroupRecords.size();
	}

	/*! render one frame */
	void SampleRenderer::render()
	{
		auto start = std::chrono::high_resolution_clock::now();
		// sanity check: make sure we launch only after first resize is
		// already done:
		if (launchParams.frame.size.x == 0)
			return;

		launchParamsBufferRender.upload(&launchParams, 1);
		launchParams.frame.accumID++;

		OPTIX_CHECK(optixLaunch(/*! pipeline we're launching launch: */
			pipeline, stream,
			/*! parameters and SBT */
			launchParamsBufferRender.d_pointer(),
			launchParamsBufferRender.sizeInBytes,
			&sbtRender,
			/*! dimensions of the launch: */
			launchParams.frame.size.x,
			launchParams.frame.size.y,
			1));

		launchParams.onlyPhotons = false;

		// sync - make sure the frame is rendered before we download and
		// display (obviously, for a high-performance application you
		// want to use streams and double-buffering, but for this simple
		// example, this will have to do)
		CUDA_SYNC_CHECK();
		auto finish = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> elapsed = finish - start;
		if (launchParams.frame.accumID <= 2)
			std::cout << "Render pass elapsed time: " << elapsed.count() << " s\n";
	}

	/*! set camera to render with */
	void SampleRenderer::setCamera(const Camera& camera)
	{
		lastSetCamera = camera;
		launchParams.camera.position = camera.from;
		launchParams.camera.direction = normalize(camera.at - camera.from);
		const float cosFovy = 0.66f;
		const float aspect = launchParams.frame.size.x / float(launchParams.frame.size.y);
		launchParams.camera.horizontal = cosFovy * aspect * normalize(cross(launchParams.camera.direction, camera.up));
		launchParams.camera.vertical = cosFovy * normalize(cross(launchParams.camera.horizontal,
			launchParams.camera.direction));
	}

	void SampleRenderer::setParams(int numPhotonSamples, int maxDepth, float radius, int antialiasingLevel)
	{
		launchParams.numPhotonSamples = numPhotonSamples;
		launchParams.maxDepth = maxDepth;
		launchParams.maxRadius = radius;
		launchParams.antialiasingLevel = antialiasingLevel;
	}

	/*! resize frame buffer to given resolution */
	void SampleRenderer::resize(const vec2i& newSize)
	{
		// if window minimized
		if (newSize.x == 0 | newSize.y == 0)
			return;

		// resize our cuda frame buffer
		colorBuffer.resize(newSize.x * newSize.y * sizeof(uint32_t));

		// update the launch parameters that we'll pass to the optix
		// launch:
		launchParams.frame.size = newSize;
		launchParams.frame.colorBuffer = (uint32_t*)colorBuffer.d_pointer();

		// and re-set the camera, since aspect may have changed
		setCamera(lastSetCamera);
	}

	/*! download the rendered color buffer */
	void SampleRenderer::downloadPixels(uint32_t h_pixels[])
	{
		colorBuffer.download(h_pixels,
			launchParams.frame.size.x * launchParams.frame.size.y);
	}

	void SampleRenderer::downloadPhotons(PhotonPrint* h_pixels)
	{
		prePhotonMap.download(h_pixels, launchParams.numPhotonSamples * launchParams.maxDepth);
	}

	/*! renders only the photons for the next frame */
	void SampleRenderer::renderPhotons()
	{
		launchParams.onlyPhotons = true;
	}

	std::string SampleRenderer::getConfigStr() {
		std::string file(launchParams.objFileName);

		//http://www.cplusplus.com/reference/string/string/find_last_of/
		//std::cout << "Splitting: " << file << '\n';
		std::size_t found = file.find_last_of("/\\");
		std::size_t ext = file.find_last_of(".");
		//std::cout << " path: " << file.substr(0, found) << '\n';
		//std::cout << " file: " << file.substr(found + 1) << '\n';

		return file.substr(found + 1)
			+ "_pos" + to_string(launchParams.light.origin.x) + to_string(launchParams.light.origin.y) + to_string(launchParams.light.origin.z)
			+ "_dir" + to_string(launchParams.light.normal.x) + to_string(launchParams.light.normal.y) + to_string(launchParams.light.normal.z)
			+ "_pwr" + to_string(launchParams.light.intensity.x) + to_string(launchParams.light.intensity.y) + to_string(launchParams.light.intensity.z)
			+ "_ps" + to_string(launchParams.numPhotonSamples)
			+ "_md" + to_string(launchParams.maxDepth)
			+ "_r" + to_string(launchParams.maxRadius)
			+ ".data";
	}

	void SampleRenderer::savePhotonMap(vector<PhotonPrint> photons, vector<int> counts, vector<int> starts) {
		std::string config = getConfigStr();
		std::string traces_path("../../maps/traces_" + config);
		std::ofstream TRACES(traces_path, std::ios::out | std::ofstream::binary);
		// https://stackoverflow.com/questions/28492517/write-and-load-vector-of-structs-in-a-binary-file-c
		typename vector<PhotonPrint>::size_type size = photons.size();
		TRACES.write((char*)&size, sizeof(size));
		TRACES.write((char*)&photons[0], photons.size() * sizeof(PhotonPrint));
		TRACES.close();

		std::string starts_path("../../maps/starts_" + config);
		std::ofstream STARTS(starts_path, std::ios::out | std::ofstream::binary);
		std::copy(starts.begin(), starts.end(), std::ostream_iterator<int>{STARTS, " "});
		STARTS.close();

		std::string counts_path("../../maps/counts_" + config);
		std::ofstream COUNTS(counts_path, std::ios::out | std::ofstream::binary);
		std::copy(counts.begin(), counts.end(), std::ostream_iterator<int>{COUNTS, " "});
		COUNTS.close();
	}

	bool SampleRenderer::loadPhotonMap(vector<PhotonPrint>& photons, vector<int>& counts, vector<int>& starts) {
		bool ok = true;
		std::string config = getConfigStr();
		// load photon traces
		std::string traces_path("../../maps/traces_" + config);
		std::ifstream TRACES(traces_path, std::ios::in | std::ifstream::binary);
		ok = ok && TRACES.good();
		if (TRACES.good()) {
			typename vector<PhotonPrint>::size_type size = 0;
			TRACES.read((char*)&size, sizeof(size));
			photons.resize(size);
			TRACES.read((char*)&photons[0], photons.size() * sizeof(PhotonPrint));
			TRACES.close();
		}

		// load photon starts
		std::string starts_path("../../maps/starts_" + config);
		std::ifstream STARTS(starts_path, std::ios::in | std::ifstream::binary);
		ok = ok && STARTS.good();
		std::istream_iterator<int> iter_starts{ STARTS };
		std::istream_iterator<int> end_starts{};
		std::copy(iter_starts, end_starts, starts.begin());
		//std::copy(iter, end, std::back_inserter(starts));
		STARTS.close();

		// load photon counts
		std::string counts_path("../../maps/counts_" + config);
		std::ifstream COUNTS(counts_path, std::ios::in | std::ifstream::binary);
		ok = ok && COUNTS.good();
		std::istream_iterator<int> iter_counts{ COUNTS };
		std::istream_iterator<int> end_counts{};
		std::copy(iter_counts, end_counts, counts.begin());
		COUNTS.close();
		return ok;
	}

} // namespace osc
