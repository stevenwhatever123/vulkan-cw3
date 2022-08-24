#version 450

layout (location = 0) in vec2 v2fTexCoord;
layout (location = 1) in vec3 iColor;
layout (location = 2) in vec3 iNormal;
layout (location = 3) in vec3 iCameraPos;
layout (location = 4) in vec3 iLightPos;
layout (location = 5) in vec3 iLightColor;
layout (location = 6) in vec3 iPosition;
layout (location = 7) in vec3 iSurfaceNormal;

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
	// Phong shading
	vec4 emissiveColor = uMaterial.emissive;

	vec4 ambientColor = vec4(0.02f, 0.02f, 0.02f, 1.0f);

	vec4 ambientMultiDiffuse = ambientColor * uMaterial.diffuse;

	vec3 lightDirection = normalize(iLightPos - iPosition);
	vec3 viewDirection = normalize(iCameraPos - iPosition);

	float clampedValue = max(0, dot(iNormal, lightDirection));
	
	vec4 diffuseDivPi = uMaterial.diffuse / radians(180);

	vec4 diffuseDivPiMultiLight = diffuseDivPi * vec4(iLightColor, 1.0f);
	
	// Compmuted from light and view directions
	vec3 halfVector = normalize(lightDirection + viewDirection);

	float shininess = ((uMaterial.shininess + 2) / 8) * 
		pow(max(0, dot(iNormal, halfVector)), uMaterial.shininess);

	vec4 specularMultiLight = clampedValue * uMaterial.specular * vec4(iLightColor, 1.0f);

	oColor = emissiveColor + ambientMultiDiffuse + (clampedValue * diffuseDivPiMultiLight)
		+ (shininess * specularMultiLight);

	// Normal
	//oColor = vec4(iNormal, 1);

	// View direction
	//oColor = vec4(viewDirection, 1);

	// Light direction
	//oColor = vec4(lightDirection, 1);
}
