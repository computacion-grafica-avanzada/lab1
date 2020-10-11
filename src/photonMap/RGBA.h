#pragma once

//No hay a
class RGBA{
public:
	float r, g, b;
public:
	RGBA();
	RGBA(float r, float g, float b);
	RGBA	operator + (const RGBA col);
	RGBA	operator - (const RGBA col);
	RGBA	operator * (const float factor);
	RGBA	operator / (const float factor);
	void clamp();
	~RGBA();
};

