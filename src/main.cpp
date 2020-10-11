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
#include "Model.h"
#include "photonMap/PhotonMap.h"
#include "photonMap/Vector3D.h"
#include "photonMap/RGBA.h"

int TOTAL_PHOTONS = 10000;

/*! \namespace osc - Optix Siggraph Course */
namespace osc {

	struct SampleWindow : public GLFCameraWindow
	{
		SampleWindow(
			const std::string& title,
			const Model* model,
			const Camera& camera,
			const std::vector<Light*> lights,
			const float worldScale)
			: GLFCameraWindow(title, camera.from, camera.at, camera.up, worldScale), sample(model)
		{
			sample.setCamera(camera);
			sample.setLights(lights);
		}

		virtual void render() override
		{
			if (cameraFrame.modified) {
				sample.setCamera(Camera{ cameraFrame.get_from(),
										 cameraFrame.get_at(),
										 cameraFrame.get_up() });
				cameraFrame.modified = false;
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


	/*! main entry point to this example - initially optix, print hello
	  world, then exit */
	extern "C" int main(int ac, char** av)
	{
		try {
			//std::vector<TriangleMesh> model(3);
			// 100x100 thin ground plane
			//model[0].color = vec3f(0.5f, 1.f, 1.f);
			//model[0].addCube(vec3f(0.f, -1.05f, 0.f), vec3f(10.f, .1f, 10.f));
			//// a unit cube centered on top of that
			//model[1].color = vec3f(1.f, 0.f, 0.f);
			//model[1].addCube(vec3f(0.f, 0.f, 0.f), vec3f(2.f, 2.f, 2.f));
			//// simulates the light
			//model[2].color = vec3f(1.f, 1.f, 1.f);
			//model[2].addCube(vec3f(2.f, 0.f, 0.f), vec3f(0.3f, 0.3f, 0.3f));
			//std::cout << gdt::vec3f(1, 2, 3) * gdt::vec3f(2,4,6) << std::endl;

			//std::vector<Photon> fotones;
			//Photon cero(Vector3D(1., 1., 1.), Vector3D(1., 0., 0.), RGBA(10.,10.,10.));
			//fotones.push_back(cero);
			//Photon uno(Vector3D(1., 2., 3.), Vector3D(1., 0., 0.), RGBA(10.,10.,10.));
			//fotones.push_back(uno);
			//Photon dos(Vector3D(5., 2., 3.), Vector3D(1., 0., 0.), RGBA(10.,10.,10.));
			//fotones.push_back(dos);
			//Photon tres(Vector3D(0., 5., 3.), Vector3D(1., 0., 0.), RGBA(10.,10.,10.));
			//fotones.push_back(tres);
			//fotones.push_back(tres);
			//fotones.push_back(tres);
			//
			//PhotonMap mapa(fotones);

			//std::cout << mapa.numPhotons() << std::endl;
			//std::vector<Photon*> *mi = mapa.getNeighbourhood(Vector3D(1, 2, 3), NULL, 1.f, 5);

			//PhotonsIterator* it = mapa.iterator();
			//int a = 0;
			//do {
			//	std::cout << a << std::endl;
			//	Photon* qw = it->next();
			//	std::cout << (*qw) << std::endl;
			//	a++;
			//} while (it->hasNext());

			//if (mi) {
			//	std::cout << "size: " << mi->size() << std::endl;
			//}
			//else {
			//	std::cout << "NULL" << std::endl;
			//}

			Model* model = loadOBJ("../../models/CornellBox-Water.obj");

			Camera camera = { 
				/*from*/vec3f(0.f,0.f,5.f),
				/* at */vec3f(0.f,0.7f,0.f),
				/* up */vec3f(0.f,1.f,0.f) 
			};

			PointLight light1(vec3f(0, 0, 0), vec3f(100));
			std::vector<Light*> lights = { &light1 };
			
			vec3f totalPower(0);
			for (int i = 0; i < lights.size(); i++) {
				totalPower += lights[i]->power;
			}

			for (int i = 0; i < lights.size(); i++) {
				vec3f scaledPower = lights[i]->power / totalPower;
				int numberPhotons = TOTAL_PHOTONS * scaledPower[arg_max(scaledPower)];
				lights[i]->setNumberPhotons(numberPhotons);
				totalPower += lights[i]->power;
			}

			// something approximating the scale of the world, so the
			// camera knows how much to move for any given user interaction:
			const float worldScale = 10.f;

			SampleWindow* window = new SampleWindow("Optix 7 Course Example",
				model, camera, lights, worldScale);

			window->run();

		}
		catch (std::runtime_error& e) {
			std::cout << GDT_TERMINAL_RED << "FATAL ERROR: " << e.what()
				<< GDT_TERMINAL_DEFAULT << std::endl;
			exit(1);
		}
		return 0;
	}

} // ::osc
