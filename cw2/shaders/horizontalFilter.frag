#version 450
#extension GL_KHR_vulkan_glsl: enable

layout (location = 0) in vec2 inUV;

layout (set = 0, binding = 0) uniform sampler2D uTexColor;

layout (location = 0) out vec4 oColor;
//layout (location = 1) out vec4 brightColor;

void main()
{
	vec3 pixelColor = texture(uTexColor, inUV).rgb;

	//oColor = vec4(pixelColor, 1.0f);

	// ==============================================
	//float weight[5] = float[] (0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216);
	float weight[23] = float[] (0.0281,	0.0281,	0.0279,	0.0277,	0.0274,	0.0270,	
	0.0266,	0.0261,	0.0255,	0.0248,	0.0241,	0.0233,	0.0225,	0.0216,	0.0208,	
	0.0199,	0.0189,	0.0180,	0.0170,	0.0161,	0.0152,	0.0142,	0.0133);

	vec2 tex_offset = 1.0 / textureSize(uTexColor, 0);
	vec3 result = texture(uTexColor, inUV).rgb * weight[0]; // Current fragment

	for(int i = 1; i < 23; i++)
	{
		// Horizontal
		result += texture(uTexColor, inUV + vec2(tex_offset.x * i, 0.0)).rgb * weight[i];
		result += texture(uTexColor, inUV - vec2(tex_offset.x * i, 0.0)).rgb * weight[i];
	}

	oColor = vec4(result, 1.0f);
}