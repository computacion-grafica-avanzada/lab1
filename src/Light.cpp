#include "Light.h"

Light::Light(vec3f _power) {
	power = _power;
}

void Light::setNumberPhotons(int number) {
	numberPhotons = number;
}

PointLight::PointLight(vec3f _position, vec3f _power) : Light(_power) {
	position = _position;
}