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
	
	mat4 rotation;
	int size;
} uScene;

layout (location = 0) out vec2 v2fTexCoord;
layout (location = 1) out vec3 oColor;
layout (location = 2) out vec3 oNormal;
layout (location = 3) out vec3 oCameraPos;
layout (location = 4) out vec3 oLightPos[3];
layout (location = 7) out vec3 oLightColor[3];
layout (location = 10) out vec3 oPosition;
layout (location = 11) out vec3 oSurfaceNormal;

void main()
{
	v2fTexCoord = texCoord;
	oColor = iColor;

	oNormal = normalize(vec3(uScene.rotation * vec4(normal, 1.0f)));
	//oNormal = normal;
	oCameraPos = vec3(uScene.camera * uScene.cameraPos);

	for(int i = 0; i < uScene.size; i++)
	{
		//oLightPos[i] = vec3(uScene.camera * uScene.lightPos[i]);
		oLightPos[i] = vec3(uScene.lightPos[i]);
		oLightColor[i] = vec3(uScene.lightColor[i]);
	}
	oPosition = vec3(uScene.camera * vec4(position, 1.0f));
	oSurfaceNormal = vec3(uScene.rotation * vec4(surfaceNormal, 1.0f));

	gl_Position = uScene.projcam * vec4(position, 1.0f);
}
