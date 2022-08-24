#version 450

layout (location = 0) in vec2 v2fTexCoord;
layout (location = 1) in vec3 iColor;
layout (location = 2) in vec3 iNormal;
layout (location = 3) in vec3 iCameraPos;
layout (location = 4) in vec3 iLightPos;
layout (location = 5) in vec3 iLightColor;
layout (location = 6) in vec3 iPosition;

layout (location = 0) out vec4 oColor;

// Blinn-Phong material (example):
layout( set = 1, binding = 0 ) uniform UMaterial
{
	vec4 emissive;
	vec4 diffuse;
	vec4 specular;
	float shininess; // Last to make std140 alignment easier.
} uMaterial;

void main()
{
	// Color
	oColor = vec4(iColor, 1.0);
}