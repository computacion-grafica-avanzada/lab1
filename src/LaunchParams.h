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

#include "gdt/math/vec.h"
#include "optix7.h"
#include "Photon.h"

#define MAX_RADIUS 0.1

const int NUM_PHOTON_SAMPLES = 100000;
const int MAX_DEPTH = 10;

namespace osc {
  using namespace gdt;

  // for this simple example, we have a single ray type
  enum { RADIANCE_RAY_TYPE=0, PHOTON_RAY_TYPE, CAUSTIC_RAY_TYPE, SHADOW_RAY_TYPE, RAY_TYPE_COUNT };

  struct TriangleMeshSBTData {
    vec3f  color;
    vec3f  specular;
    vec3f  transmission;
    vec3f *vertex;
    vec3f *normal;
    vec2f *texcoord;
    vec3i *index;
    bool                hasTexture;
    cudaTextureObject_t texture;
    float ior, phong;
  };

  struct PhotonPrint {
      vec3f position;
      vec3f power;
      bool operator==(PhotonPrint const& other) {
          return position == other.position && power == other.power;
      }
      bool operator!=(PhotonPrint const& other) {
          return position != other.position || power != other.power;
      }
  };

  struct LaunchParams
  {
    struct {
      uint32_t *colorBuffer;
      vec2i     size;
      int       accumID { 0 };
    } frame;

    struct {
      vec3f position;
      vec3f direction;
      vec3f horizontal;
      vec3f vertical;
    } camera;

    struct {
      vec3f origin, normal, intensity, photonPower;
    } light;

    vec2f* halton;

    PhotonPrint* prePhotonMap;
    PhotonPrint* photonMap;
    int mapSize;

    // hash grid
    PhotonPrint* pm;
    int* pmCount;
    int* pmStarts;
    vec3f gridSize;
    vec3f lowerBound;
    
    OptixTraversableHandle traversable;
  };

} // ::osc
