#version 450

layout (location = 0) in vec2 v2fTexCoord;
layout (location = 1) in vec3 iColor;
layout (location = 2) in vec3 iNormal;
layout (location = 3) in vec3 iCameraPos;
layout (location = 4) in vec3 iLightPos[3];
layout (location = 7) in vec3 iLightColor[3];
layout (location = 10) in vec3 iPosition;
layout (location = 11) in vec3 iSurfaceNormal;

layout (location = 0) out vec4 oColor;

// Blinn-Phong material (example):
layout( set = 1, binding = 0 ) uniform UMaterial
{
	vec4 emissive;
	vec4 diffuse;
	vec4 specular;
	float shininess; // Last to make std140 alignment easier.
} uMaterial;

// PBR material (example):
layout( set = 2, binding = 0 ) uniform PBRMaterial
{
	vec4 emissive;
	vec4 albedo;
	float shininess;
	float metalness;
	int size;
} pbrMaterial;

void main()
{
	vec4 ambientColor = vec4(0.02f, 0.02f, 0.02f, 1.0f);

	vec3 brdfSum = vec3(0, 0, 0);

	// Loop through every light source
	for(int i = 0; i < pbrMaterial.size; i++)
	{
		vec3 lightDirection = normalize(iLightPos[i] - iPosition);
		vec3 viewDirection = normalize(iCameraPos - iPosition);
		vec3 halfVector = normalize(lightDirection + viewDirection);

		// Fresnel calculation - F
		vec3 f0 = (1 - pbrMaterial.metalness) * vec3(0.04f, 0.04f, 0.04f) + (pbrMaterial.metalness * iColor);
		vec3 fresnel = f0 + (1 - f0) * pow((1 - dot(halfVector, viewDirection)), 5);

		vec3 diffuse = (iColor / radians(180)) * (vec3(1, 1, 1) - fresnel) * (1 - pbrMaterial.metalness);

	
		// normal distribution - D
		float normalDistribution = ((pbrMaterial.shininess + 2) / (2 * radians(180)))
								 * pow(max(0, dot(iNormal, halfVector)), pbrMaterial.shininess);
		
		// Our custom one
		// Kelemen
//		float viewHalfDot = dot(viewDirection, halfVector);
//		if(isnan(viewHalfDot))
//			viewHalfDot = 0.000001;
//		float normalDistribution = (dot(iNormal, lightDirection) * dot(iNormal, viewDirection))
//									/ (viewHalfDot * viewHalfDot);

		// Cook-Torrance model calculations - G
		float leftTop = 2 * max(0, dot(iNormal, halfVector)) * max(0, dot(iNormal, viewDirection));
		float leftBottom = dot(viewDirection, halfVector);
		if(isnan(leftBottom))
			leftBottom = 0.0000001f;
		float leftModel = leftTop / leftBottom;

		float rightTop = 2 * max(0, dot(iNormal, halfVector)) * max(0, dot(iNormal, lightDirection));
		float rightBottom = dot(viewDirection, halfVector);
		if(isnan(rightBottom))
			rightBottom = 0.0000001f;
		float rightModel = rightTop / rightBottom;

		float cookTorranceModel = min(1, min(leftModel, rightModel));

		// BRDF
		vec3 frTop = (normalDistribution * fresnel * cookTorranceModel);
		float frBottom = 4 * max(0, dot(iNormal, viewDirection)) * max(0, dot(iNormal, lightDirection));
		frBottom = frBottom + 0.000001f;

		vec3 fr = diffuse + (frTop / frBottom);

		vec3 result = (fr * iLightColor[i]) * max(0, dot(iNormal, lightDirection));

		brdfSum = brdfSum + result;
		//brdfSum = vec3(normalDistribution);
	}

	// Ambient
	vec3 ambient = vec3(ambientColor) * iColor;

	// emissive
	vec3 emissive = vec3(pbrMaterial.emissive) * iColor;

	vec4 pixelColor = vec4(emissive + ambient + brdfSum, 1);

	float sum = pixelColor.x + pixelColor.y + pixelColor.z;

	if(sum >= 1.0f)
	{
		oColor = pixelColor;
	}
	else
	{
		oColor = vec4(0, 0, 0, 1);
	}
}