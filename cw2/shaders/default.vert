#version 450

layout (location = 0) in vec3 position;
layout (location = 1) in vec3 normal;
layout (location = 2) in vec2 texCoord;
layout (location = 3) in vec3 iColor;
layout (location = 4) in vec3 surfaceNormal;

layout(set = 0, binding = 0) uniform UScene
{
	mat4 camera;
	mat4 projection;
	mat4 projcam;

	vec4 cameraPos;
	vec4 lightPos[3];
	vec4 lightColor[3];
	float size;
} uScene;

layout (location = 0) out vec2 v2fTexCoord;
layout (location = 1) out vec3 oColor;
layout (location = 2) out vec3 oNormal;
layout (location = 3) out vec3 oCameraPos;
layout (location = 4) out vec3 oLightPos;
layout (location = 5) out vec3 oLightColor;
layout (location = 6) out vec3 oPosition;

void main()
{
	v2fTexCoord = texCoord;
	oColor = iColor;

	oNormal = normal;
	oCameraPos = vec3(uScene.cameraPos);
	oLightPos = vec3(uScene.camera * uScene.lightPos[0]);
	oLightColor = vec3(uScene.lightColor[0]);
	oPosition = vec3(uScene.camera * vec4(position, 1.0f));

	gl_Position = uScene.projcam * vec4(position, 1.0f);
}
