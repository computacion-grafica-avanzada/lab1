#pragma once
#include "gdt/math/vec.h"

using namespace gdt;

class Light {
public:
	vec3f power;			// Watts
	int numberPhotons;	// relative to other lights
	Light(vec3f power);
	void setNumberPhotons(int number);
};

class PointLight : public Light {
public:
	vec3f position;
	PointLight(vec3f position, vec3f power);
};

// todo other lights