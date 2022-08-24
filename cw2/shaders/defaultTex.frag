#version 450
#extension GL_KHR_vulkan_glsl : enable

layout (location = 0) in vec2 v2fTexCoord;
layout (location = 1) in vec3 iColor;
layout (location = 2) in vec3 iNormal;
layout (location = 3) in vec3 iCameraPos;
layout (location = 4) in vec3 iLightPos;
layout (location = 5) in vec3 iLightColor;
layout (location = 6) in vec3 iPosition;

layout (set = 1, binding = 0) uniform sampler2D uTexColor;

layout (location = 0) out vec4 oColor;

void main()
{
	oColor = vec4(texture(uTexColor, v2fTexCoord).rgb, 1.0f);
}