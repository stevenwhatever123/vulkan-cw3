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
	vec4 lightPos;
	vec4 lightColor;
	
	mat4 rotation;
} uScene;

layout (location = 0) out vec2 v2fTexCoord;
layout (location = 1) out vec3 oColor;
layout (location = 2) out vec3 oNormal;
layout (location = 3) out vec3 oCameraPos;
layout (location = 4) out vec3 oLightPos;
layout (location = 5) out vec3 oLightColor;
layout (location = 6) out vec3 oPosition;
layout (location = 7) out vec3 oSurfaceNormal;

void main()
{
	v2fTexCoord = texCoord;
	oColor = iColor;

	oNormal = normalize(vec3(uScene.rotation * vec4(normal, 1.0f)));
	oCameraPos = vec3(uScene.camera * uScene.cameraPos);
	oLightPos = vec3(uScene.camera * uScene.lightPos);
	oLightColor = vec3(uScene.lightColor);
	oPosition = vec3(uScene.camera * vec4(position, 1.0f));
	oSurfaceNormal = vec3(uScene.rotation * vec4(surfaceNormal, 1.0f));

	gl_Position = uScene.projcam * vec4(position, 1.0f);
}
