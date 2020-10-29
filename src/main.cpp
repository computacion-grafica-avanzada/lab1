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

// our helper library for window handling
#include "glfWindow/GLFWindow.h"
#include <GL/gl.h>

#include "PhotonMap.h"
#include "halton_seq.h"
#include <ctime>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "3rdParty/stb_image_write.h"
#include <chrono>




/*! \namespace osc - Optix Siggraph Course */
namespace osc {

	int drawLoops = 0;

	struct SampleWindow : public GLFCameraWindow
	{
		SampleWindow(const std::string& title,
			const Model* model,
			const Camera& camera,
			const PointLight& light,
			const float worldScale,
			std::string objFileName)
			: GLFCameraWindow(title, camera.from, camera.at, camera.up, worldScale), sample(model, light, objFileName)
		{
			sample.setCamera(camera);
		}

		virtual void render() override
		{
			if (cameraFrame.modified) {
				sample.setCamera(Camera{ cameraFrame.get_from(),
										 cameraFrame.get_at(),
										 cameraFrame.get_up() });
				cameraFrame.modified = false;
			}

			if (cameraFrame.screenshot)
			{
				cameraFrame.screenshot = false;
				drawLoops = 0;
				sample.renderPhotons();
			}

			sample.render();
		}

		virtual void draw() override
		{
			sample.downloadPixels(pixels.data());
			if (fbTexture == 0)
				glGenTextures(1, &fbTexture);

			glBindTexture(GL_TEXTURE_2D, fbTexture);
			GLenum texFormat = GL_RGBA;
			GLenum texelType = GL_UNSIGNED_BYTE;
			glTexImage2D(GL_TEXTURE_2D, 0, texFormat, fbSize.x, fbSize.y, 0, GL_RGBA,
				texelType, pixels.data());

			glDisable(GL_LIGHTING);
			glColor3f(1, 1, 1);

			glMatrixMode(GL_MODELVIEW);
			glLoadIdentity();

			glEnable(GL_TEXTURE_2D);
			glBindTexture(GL_TEXTURE_2D, fbTexture);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

			glDisable(GL_DEPTH_TEST);

			glViewport(0, 0, fbSize.x, fbSize.y);

			glMatrixMode(GL_PROJECTION);
			glLoadIdentity();
			glOrtho(0.f, (float)fbSize.x, 0.f, (float)fbSize.y, -1.f, 1.f);

			glBegin(GL_QUADS);
			{
				glTexCoord2f(0.f, 0.f);
				glVertex3f(0.f, 0.f, 0.f);

				glTexCoord2f(0.f, 1.f);
				glVertex3f(0.f, (float)fbSize.y, 0.f);

				glTexCoord2f(1.f, 1.f);
				glVertex3f((float)fbSize.x, (float)fbSize.y, 0.f);

				glTexCoord2f(1.f, 0.f);
				glVertex3f((float)fbSize.x, 0.f, 0.f);
			}
			glEnd();

			if (drawLoops <= 1)
			{
				time_t now = time(0);
				tm* ltm = new tm();
				localtime_s(ltm, &now);
				std::stringstream ss, ssBmp;

				if (drawLoops == 0)
				{
					ss << "../../results/PhotonMap" << 1 + ltm->tm_mon << "_" << ltm->tm_mday << " " << ltm->tm_hour << "_" << ltm->tm_min << "_" << ltm->tm_sec;
					ssBmp << ss.str() << ".png";
				}
				else
				{
					ss << "../../results/Image" << 1 + ltm->tm_mon << "_" << ltm->tm_mday << " " << ltm->tm_hour << "_" << ltm->tm_min << "_" << ltm->tm_sec;
					ssBmp << ss.str() << ".png";
				}

				std::vector<uint32_t> pixels_r;

				stbi_flip_vertically_on_write(1);

				stbi_write_png(ssBmp.str().c_str(), fbSize.x, fbSize.y, 4,
					pixels.data(), fbSize.x * sizeof(uint32_t));
			}
			drawLoops++;
		}

		virtual void resize(const vec2i& newSize)
		{
			fbSize = newSize;
			sample.resize(newSize);
			pixels.resize(newSize.x * newSize.y);
		}

		vec2i                 fbSize;
		GLuint                fbTexture{ 0 };
		SampleRenderer        sample;
		std::vector<uint32_t> pixels;
	};

	float readFloat(FILE* file)
	{
		string number_s;
		char character;

		character = fgetc(file);
		while (character != ' ' && character != '\n' && character != EOF)
		{
			number_s.push_back(character);
			character = fgetc(file);
		}

		return atof(number_s.c_str());
	}

	int readInt(FILE* file)
	{
		string number_s;
		char character;

		character = fgetc(file);
		while (character != ' ' && character != '\n' && character != EOF)
		{
			number_s.push_back(character);
			character = fgetc(file);
		}

		return atoll(number_s.c_str());
	}

	void loadParams(string& objFileName, vec3f& cameraPos, vec3f& cameraUp, vec3f& lightPos, vec3f& lightDir, vec3f& lightPower,
		int& numPhotonSamples, int& maxDepth, float& radius, int& antialiasingLevel)
	{
		char character;
		FILE* file;
		file = fopen("../../params.txt", "r");

		if (file) {
			// Read file name
			while ((character = fgetc(file)) != '\n' && character != EOF)
			{
				objFileName.push_back(character);
			}

			// Read camera pos
			cameraPos.x = readFloat(file);
			cameraPos.y = readFloat(file);
			cameraPos.z = readFloat(file);

			// Read camera up
			cameraUp.x = readFloat(file);
			cameraUp.y = readFloat(file);
			cameraUp.z = readFloat(file);

			// Read light pos
			lightPos.x = readFloat(file);
			lightPos.y = readFloat(file);
			lightPos.z = readFloat(file);

			// Read light dir
			lightDir.x = readFloat(file);
			lightDir.y = readFloat(file);
			lightDir.z = readFloat(file);

			// Read light power
			lightPower.x = readFloat(file);
			lightPower.y = readFloat(file);
			lightPower.z = readFloat(file);

			// Read number of photons
			numPhotonSamples = readInt(file);

			// Read max depth
			maxDepth = readInt(file);

			// Read radius
			radius = readFloat(file);

			// Read antialiasing level
			antialiasingLevel = readInt(file);

			fclose(file);
		}
		else
		{
			printf("ERROR al intentar abrir archivo de parametros!");
		}
	}


	/*! main entry point to this example - initially optix, print hello
	  world, then exit */
	extern "C" int main(int ac, char** av)
	{
		try {
			auto start = std::chrono::high_resolution_clock::now();
			string objFileName;
			vec3f cameraPos;
			vec3f cameraUp;
			vec3f lightPos;
			vec3f lightDir;
			vec3f lightPower;
			int numPhotonSamples;
			int maxDepth;
			float radius;
			int antialiasingLevel;

			loadParams(objFileName, cameraPos, cameraUp, lightPos, lightDir, lightPower, numPhotonSamples, maxDepth, radius, antialiasingLevel);

			Model* model = loadOBJ(objFileName);

			Camera camera = {
				/*from*/cameraPos,
				/* at */model->bounds.center(),
				/* up */cameraUp
			};

			PointLight light = { numPhotonSamples, lightPos, lightDir, lightPower };

			// something approximating the scale of the world, so the
			// camera knows how much to move for any given user interaction:
			const float worldScale = length(model->bounds.span());

			SampleWindow* window = new SampleWindow("Optix 7 Course Example", model, camera, light, worldScale, objFileName);
			window->sample.setParams(numPhotonSamples, maxDepth, radius, antialiasingLevel);
			
			auto finish = std::chrono::high_resolution_clock::now();
			std::chrono::duration<double> elapsed = finish - start;
			std::cout << "Pre-render (includes photon pass) elapsed time: " << elapsed.count() << " s\n";
			window->run();
		}
		catch (std::runtime_error& e) {
			std::cout << GDT_TERMINAL_RED << "FATAL ERROR: " << e.what()
				<< GDT_TERMINAL_DEFAULT << std::endl;
			std::cout << "Did you forget to copy sponza.obj and sponza.mtl into your optix7course/models directory?" << std::endl;
			exit(1);
		}
		return 0;
	}

} // ::osc
