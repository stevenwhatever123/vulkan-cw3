#include <volk/volk.h>

#include <tuple>
#include <chrono>
#include <limits>
#include <vector>
#include <stdexcept>

#include <cstdio>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#if !defined(GLM_FORCE_RADIANS)
#	define GLM_FORCE_RADIANS
#endif
#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/euler_angles.hpp>

#include "../labutils/to_string.hpp"
#include "../labutils/vulkan_window.hpp"

#include "../labutils/angle.hpp"
using namespace labutils::literals;

#include "../labutils/error.hpp"
#include "../labutils/vkutil.hpp"
#include "../labutils/vkimage.hpp"
#include "../labutils/vkobject.hpp"
#include "../labutils/vkbuffer.hpp"
#include "../labutils/allocator.hpp" 
namespace lut = labutils;

#include "model.hpp"

namespace
{
	namespace cfg
	{
		// Compiled shader code for the graphics pipeline(s)
		// See sources in cw1/shaders/*. 
#		define SHADERDIR_ "assets/cw2/shaders/"
		constexpr char const* kVertShaderPath = SHADERDIR_ "PBR.vert.spv";
		constexpr char const* kFragShaderPath = SHADERDIR_ "PBR.frag.spv";
		constexpr char const* kFragTexShaderPath = SHADERDIR_ "defaultTex.frag.spv";

		constexpr char const* kfilterBrightVertPath = SHADERDIR_ "filterBright.vert.spv";
		constexpr char const* kfilterBrightFragPath = SHADERDIR_ "filterBright.frag.spv";

		constexpr char const* kHorizontalFilterVertPath = SHADERDIR_ "horizontalFilter.vert.spv";
		constexpr char const* kHorizontalFilterFragPath = SHADERDIR_ "horizontalFilter.frag.spv";

		constexpr char const* kVerticalFilterVertPath = SHADERDIR_ "verticalFilter.vert.spv";
		constexpr char const* kVerticalFilterFragPath = SHADERDIR_ "verticalFilter.frag.spv";

		constexpr char const* kPostProcessinVertgPath = SHADERDIR_ "post.vert.spv";
		constexpr char const* kPostProcessingFragPath = SHADERDIR_ "post.frag.spv";
#		undef SHADERDIR_

#		define MODELDIR_ "assets/cw3/"
		constexpr char const* kShipPath = MODELDIR_ "NewShip.obj";
		constexpr char const* kMaterialTestPath = MODELDIR_ "materialtest.obj";
#		undef MODELDIR_

		constexpr VkFormat kDepthFormat = VK_FORMAT_D32_SFLOAT;


		// General rule: with a standard 24 bit or 32 bit float depth buffer,
		// you can support a 1:1000 ratio between the near and far plane with
		// minimal depth fighting. Larger ratios will introduce more depth
		// fighting problems; smaller ratios will increase the depth buffer's
		// resolution but will also limit the view distance.
		constexpr float kCameraNear  = 0.1f;
		constexpr float kCameraFar   = 100.f;

		constexpr auto kCameraFov    = 60.0_degf;
	}


	// Local types/structures:

	// Local functions:

	// GLFW callbacks
	void glfw_callback_key_press(GLFWwindow*, int, int, int, int);
	void glfw_callback_mouse_press(GLFWwindow*, int, int, int);
	void glfw_callback_mouse_position(GLFWwindow*, double, double);
	double mouseX, mouseY;

	namespace glsl
	{
		struct SceneUniform
		{
			glm::mat4 camera;
			glm::mat4 projection;
			glm::mat4 projcam;

			glm::vec4 cameraPos;
			glm::vec4 lightPos[3];
			glm::vec4 lightColor[3];
			glm::mat4 rotation;
			int size = 3;
		};

		static_assert(sizeof(SceneUniform) <= 65536,
			"SceneUniform must be less than 65536 bytes for vkCmdUpdateBuffer");
		static_assert(sizeof(SceneUniform) % 4 == 0,
			"SceneUniform size must be multiple of 4 bytes");

		struct MaterialUniform
		{
			glm::vec4 emissive;
			glm::vec4 diffuse;
			glm::vec4 specular;
			float shininess;
		};

		static_assert(sizeof(MaterialUniform) <= 65536,
			"MaterialUniform must be less than 65536 bytes for vkCmdUpdateBuffer");
		static_assert(sizeof(MaterialUniform) % 4 == 0,
			"MaterialUniform size must be multiple of 4 bytes");

		struct MaterialPBRUniform
		{
			glm::vec4 emissive;
			glm::vec4 albedo;
			float shininess;
			float metalness;
			int size = 3;
		};

		static_assert(sizeof(MaterialPBRUniform) <= 65536,
			"MaterialPBRUniform must be less than 65536 bytes for vkCmdUpdateBuffer");
		static_assert(sizeof(MaterialPBRUniform) % 4 == 0,
			"MaterialPBRUniform size must be multiple of 4 bytes");
	}

	// Camera Position
	glm::vec3 position(0.0f, 0.0f, -5.0f);
	glm::vec3 rotation(0);
	int multiplier = 5;
	bool moveCamera = false;
	int numLight = 1;

	lut::RenderPass create_render_pass(lut::VulkanWindow const&);
	lut::RenderPass create_render_pass_texture(lut::VulkanWindow const&);

	lut::DescriptorSetLayout create_scene_descriptor_layout(lut::VulkanWindow const&);
	lut::DescriptorSetLayout create_material_descriptor_layout(lut::VulkanWindow const&);
	lut::DescriptorSetLayout create_object_descriptor_layout(lut::VulkanWindow const&);

	lut::PipelineLayout create_pipeline_layout(lut::VulkanContext const&, VkDescriptorSetLayout, 
		VkDescriptorSetLayout, VkDescriptorSetLayout);
	lut::PipelineLayout create_postprocess_pipeline_layout(lut::VulkanContext const&, VkDescriptorSetLayout, VkDescriptorSetLayout);
	
	lut::Pipeline create_pipeline(lut::VulkanWindow const&, VkRenderPass, VkPipelineLayout);
	lut::Pipeline create_pipeline_filter_bright(lut::VulkanWindow const&, VkRenderPass, VkPipelineLayout);

	lut::PipelineLayout create_pipeline_with_texture_layout(lut::VulkanContext const&, VkDescriptorSetLayout, VkDescriptorSetLayout);
	lut::Pipeline create_pipeline_with_texture(lut::VulkanWindow const&, VkRenderPass, VkPipelineLayout);
	lut::Pipeline create_postprocess_pipeline(lut::VulkanWindow const&, VkRenderPass, VkPipelineLayout);
	lut::Pipeline create_pipeline_horizontal(lut::VulkanWindow const&, VkRenderPass, VkPipelineLayout);
	lut::Pipeline create_pipeline_vertical(lut::VulkanWindow const&, VkRenderPass, VkPipelineLayout);

	void create_swapchain_framebuffers(
		lut::VulkanWindow const&,
		VkRenderPass,
		std::vector<lut::Framebuffer>&,
		VkImageView
	);

	void create_framebuffer(
		lut::VulkanWindow const&,
		VkRenderPass,
		lut::Framebuffer&,
		VkImageView,
		VkImageView 
	); 
	 
	void create_frame_texture(
		lut::VulkanWindow const&,
		VkRenderPass,
		lut::Framebuffer&,
		VkImageView,
		VkImageView
	);

	void update_scene_uniforms(
		glsl::SceneUniform&,
		std::uint32_t aFramebufferWidth,
		std::uint32_t aFramebufferHeight,
		int numLight
	);

	void update_material_uniforms(
		glsl::MaterialUniform& aMaterialUniforms,
		glm::vec4 emissive,
		glm::vec4 diffuse,
		glm::vec4 specular,
		float shininess
	);

	void update_material_PBR_uniforms(
		glsl::MaterialPBRUniform& aMaterialUniforms,
		glm::vec4 emissive,
		glm::vec4 albedo,
		float shininess,
		float metalness,
		int numLight
	);

	void record_commands(
		VkCommandBuffer,
		VkRenderPass,
		VkRenderPass,
		VkFramebuffer,
		VkFramebuffer,
		VkFramebuffer,
		VkFramebuffer,
		VkFramebuffer,
		VkPipeline,
		VkPipeline,
		VkPipeline,
		VkPipeline,
		VkPipeline,
		VkExtent2D const&,
		LoadedMesh& car,
		VkBuffer aSceneUBO,
		glsl::SceneUniform const&,
		VkPipelineLayout,
		VkPipelineLayout,
		VkDescriptorSet aSceneDescriptors,
		VkDescriptorSet aBackFrameBufferDescriptors,
		VkDescriptorSet aBackBufferDescriptor,
		VkDescriptorSet aFilterHorizontalDescriptor,
		VkDescriptorSet aFilterVerticalDescriptor,
		std::vector<labutils::Buffer> const& aMaterialUBOs,
		std::vector<glsl::MaterialUniform> const& aMaterialUniforms,
		std::vector<VkDescriptorSet>const& aMaterialDescriptors,
		std::vector<labutils::Buffer> const& aMaterialPBRUBOs,
		std::vector<glsl::MaterialPBRUniform> const& aMaterialPBRUniforms,
		std::vector<VkDescriptorSet>const& aMaterialPBRDescriptors
	);

	void post_processing(
		VkCommandBuffer,
		VkRenderPass,
		VkFramebuffer,
		VkPipeline,
		VkExtent2D const&
	);

	void submit_commands(
		lut::VulkanContext const&,
		VkCommandBuffer,
		VkFence,
		VkSemaphore,
		VkSemaphore
	);

	std::tuple<lut::Image, lut::ImageView> create_offine_image_view(lut::VulkanWindow const&, lut::Allocator const&);

	std::tuple<lut::Image, lut::ImageView> create_depth_buffer(lut::VulkanWindow const&, lut::Allocator const&);

	void updateBackBufferDescriptorSet(lut::VulkanWindow const&, VkDescriptorSet const&,
		VkImageView const&, VkSampler const&);
}

int main() try
{
	// Create vulkan window
	auto window = lut::make_vulkan_window();

	// Create VMA allocator
	lut::Allocator allocator = lut::create_allocator(window);

	// Intialize resources
	lut::RenderPass renderPass = create_render_pass(window);
	lut::RenderPass offlineRenderPass = create_render_pass_texture(window);
	
	lut::DescriptorSetLayout sceneLayout = create_scene_descriptor_layout(window);

	lut::DescriptorSetLayout materialLayout = create_material_descriptor_layout(window);

	lut::DescriptorSetLayout objectLayout = create_object_descriptor_layout(window);

	lut::PipelineLayout pipeLayout = create_pipeline_layout(window, sceneLayout.handle, materialLayout.handle, 
		objectLayout.handle);
	//lut::Pipeline pipe = create_pipeline(window, renderPass.handle, pipeLayout.handle);
	lut::Pipeline pipe = create_pipeline(window, offlineRenderPass.handle, pipeLayout.handle);
	lut::Pipeline pipe_filter_bright = create_pipeline_filter_bright(window, offlineRenderPass.handle, pipeLayout.handle);

	lut::PipelineLayout pipeLayoutTex = create_pipeline_with_texture_layout(window, sceneLayout.handle, 
		objectLayout.handle);
	//lut::Pipeline pipeTex = create_pipeline_with_texture(window, renderPass.handle, pipeLayoutTex.handle);

	lut::PipelineLayout postPipeLayout = create_postprocess_pipeline_layout(window, sceneLayout.handle, objectLayout.handle);
	lut::Pipeline postPipe = create_postprocess_pipeline(window, renderPass.handle, postPipeLayout.handle);
	lut::Pipeline filterHorizontalPipe = create_pipeline_horizontal(window, renderPass.handle, postPipeLayout.handle);
	lut::Pipeline filterVerticalPipe = create_pipeline_vertical(window, renderPass.handle, postPipeLayout.handle);

	// Depth Buffer
	auto [depthBuffer, depthBufferView] = create_depth_buffer(window, allocator);

	auto [backBuffer, backBufferView] = create_offine_image_view(window, allocator);
	auto [backFrameBuffer, backFrameBufferView] = create_offine_image_view(window, allocator);
	auto [backBufferHorizontal, backBufferViewHorizontal] = create_offine_image_view(window, allocator);
	auto [backBufferVertical, backBufferViewVertical] = create_offine_image_view(window, allocator);

	std::vector<lut::Framebuffer> framebuffers;
	create_swapchain_framebuffers(window, renderPass.handle, framebuffers, depthBufferView.handle);

	lut::CommandPool cpool = lut::create_command_pool(window, VK_COMMAND_POOL_CREATE_TRANSIENT_BIT | VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);

	std::vector<VkCommandBuffer> cbuffers;
	std::vector<lut::Fence> cbfences;

	for (std::size_t i = 0; i < framebuffers.size(); ++i)
	{
		cbuffers.emplace_back(lut::alloc_command_buffer(window, cpool.handle));
		cbfences.emplace_back(lut::create_fence(window, VK_FENCE_CREATE_SIGNALED_BIT));
	}

	lut::Semaphore imageAvailable = lut::create_semaphore(window);
	lut::Semaphore renderFinished = lut::create_semaphore(window);

	// Create descriptor pool
	lut::DescriptorPool dpool = lut::create_descriptor_pool(window);

	// Load the model data
	ModelData carModel = load_obj_model(cfg::kShipPath);
	//ModelData carModel = load_obj_model(cfg::kMaterialTestPath);
	//ModelData cityModel = load_obj_model(cfg::kMaterialTestPath);
	LoadedMesh loadedModel = create_loaded_mesh (window, allocator, dpool, objectLayout, carModel, false);

	// Create a new framebuffer for offscreen rendering
	lut::Framebuffer backFramebuffer;
	create_framebuffer(window, offlineRenderPass.handle,
		backFramebuffer, depthBufferView.handle, backFrameBufferView.handle);
	lut::Framebuffer temp_framebuffer;
	create_framebuffer(window, offlineRenderPass.handle,
		temp_framebuffer, depthBufferView.handle, backBufferView.handle);
	lut::Framebuffer temp_framebuffer_horizontal;
	create_framebuffer(window, offlineRenderPass.handle,
		temp_framebuffer_horizontal, depthBufferView.handle, backBufferViewHorizontal.handle);
	lut::Framebuffer temp_framebuffer_vertical;
	create_framebuffer(window, offlineRenderPass.handle,
		temp_framebuffer_vertical, depthBufferView.handle, backBufferViewVertical.handle);

	// Create scene uniform buffer
	lut::Buffer sceneUBO = lut::create_buffer(
		allocator,
		sizeof(glsl::SceneUniform),
		VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		VMA_MEMORY_USAGE_GPU_ONLY
	);

	// Allocate descriptor set for uniform buffer
	// Initialise descriptor set with vkUpdateDescriptorSets
	VkDescriptorSet sceneDescriptor = lut::alloc_desc_set(window, dpool.handle, sceneLayout.handle);

	{
		VkWriteDescriptorSet desc[1]{};

		VkDescriptorBufferInfo sceneUboInfo{};
		sceneUboInfo.buffer = sceneUBO.buffer;
		sceneUboInfo.range = VK_WHOLE_SIZE;

		desc[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		desc[0].dstSet = sceneDescriptor;
		desc[0].dstBinding = 0;
		desc[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		desc[0].descriptorCount = 1;
		desc[0].pBufferInfo = &sceneUboInfo;

		constexpr auto numSets = sizeof(desc) / sizeof(desc[0]);
		vkUpdateDescriptorSets(window.device, numSets, desc, 0, nullptr);
	}

	// Default sampler
	lut::Sampler filterSampler = lut::create_anisotropic_filter_sampler(window, 1);

	VkDescriptorSet backFrameBufferDescriptor = lut::alloc_desc_set(window, dpool.handle, objectLayout.handle);
	updateBackBufferDescriptorSet(window, backFrameBufferDescriptor, backFrameBufferView.handle, filterSampler.handle);

	// Back Buffer texture image
	VkDescriptorSet backBufferDescriptor = lut::alloc_desc_set(window, dpool.handle, objectLayout.handle);
	updateBackBufferDescriptorSet(window, backBufferDescriptor, backBufferView.handle, filterSampler.handle);

	// Back Buffer texture image
	VkDescriptorSet backBufferBrightHorizontal = lut::alloc_desc_set(window, dpool.handle, objectLayout.handle);
	updateBackBufferDescriptorSet(window, backBufferBrightHorizontal, backBufferViewHorizontal.handle,
		filterSampler.handle);

	VkDescriptorSet backBufferBrightVertical = lut::alloc_desc_set(window, dpool.handle, objectLayout.handle);
	updateBackBufferDescriptorSet(window, backBufferBrightVertical, backBufferViewVertical.handle,
		filterSampler.handle);
	
	
	std::vector<lut::Buffer> materialUBO(carModel.materials.size());
	// Create material uniform buffer
	for (size_t i = 0; i < materialUBO.size(); i++)
	{
		materialUBO[i] = lut::create_buffer(
			allocator,
			sizeof(glsl::MaterialUniform),
			VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
			VMA_MEMORY_USAGE_GPU_ONLY
		);
	}

	std::vector<VkDescriptorSet> materialDescriptors(carModel.materials.size());

	for (size_t i = 0; i < materialDescriptors.size(); i++)
	{
		materialDescriptors[i] = lut::alloc_desc_set(window, dpool.handle, materialLayout.handle);

		{
			VkWriteDescriptorSet desc[1]{};

			VkDescriptorBufferInfo materialUboInfo{};
			materialUboInfo.buffer = materialUBO[i].buffer;
			materialUboInfo.range = VK_WHOLE_SIZE;

			desc[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			desc[0].dstSet = materialDescriptors[i];
			desc[0].dstBinding = 0;
			desc[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
			desc[0].descriptorCount = 1;
			desc[0].pBufferInfo = &materialUboInfo;

			constexpr auto numSets = sizeof(desc) / sizeof(desc[0]);
			vkUpdateDescriptorSets(window.device, numSets, desc, 0, nullptr);
		}
	}

	std::vector<lut::Buffer> materialPBRUBO(carModel.materials.size());
	for (size_t i = 0; i < materialPBRUBO.size(); i++)
	{
		materialPBRUBO[i] = lut::create_buffer(
			allocator,
			sizeof(glsl::MaterialPBRUniform),
			VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
			VMA_MEMORY_USAGE_GPU_ONLY
		);
	}

	std::vector<VkDescriptorSet> materialPBRDescriptors(carModel.materials.size());
	for (size_t i = 0; i < materialPBRDescriptors.size(); i++)
	{
		materialPBRDescriptors[i] = lut::alloc_desc_set(window, dpool.handle, materialLayout.handle);

		{
			VkWriteDescriptorSet desc[1]{};

			VkDescriptorBufferInfo materialPBRUboInfo{};
			materialPBRUboInfo.buffer = materialPBRUBO[i].buffer;
			materialPBRUboInfo.range = VK_WHOLE_SIZE;

			desc[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			desc[0].dstSet = materialPBRDescriptors[i];
			desc[0].dstBinding = 0;
			desc[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
			desc[0].descriptorCount = 1;
			desc[0].pBufferInfo = &materialPBRUboInfo;

			constexpr auto numSets = sizeof(desc) / sizeof(desc[0]);
			vkUpdateDescriptorSets(window.device, numSets, desc, 0, nullptr);
		}
	}

	// Application main loop
	bool recreateSwapchain = false;

	while (!glfwWindowShouldClose(window.window))
	{
		glfwPollEvents();

		glfwSetKeyCallback(window.window, glfw_callback_key_press);
		glfwSetMouseButtonCallback(window.window, glfw_callback_mouse_press);
		glfwSetCursorPosCallback(window.window, glfw_callback_mouse_position);

		// Recreate swap chain?
		if (recreateSwapchain)
		{
			//TODO: re-create swapchain and associated resources!
			// We need to destroy several objects, which may still be in
			// use by the GPU. Therefore wait for the GPU
			// to finish processing
			vkDeviceWaitIdle(window.device);

			// Recreate them
			auto const changes = recreate_swapchain(window);

			if (changes.changedFormat)
				renderPass = create_render_pass(window);

			if (changes.changedSize)
			{
				std::tie(depthBuffer, depthBufferView) = create_depth_buffer(window, allocator);
				std::tie(backBuffer, backBufferView) = create_offine_image_view(window, allocator);
				std::tie(backFrameBuffer, backFrameBufferView) = create_offine_image_view(window, allocator);
				std::tie(backBufferHorizontal, backBufferViewHorizontal) = create_offine_image_view(window, allocator);
				std::tie(backBufferVertical, backBufferViewVertical) = create_offine_image_view(window, allocator);
			}
			create_framebuffer(window, offlineRenderPass.handle,
				backFramebuffer, depthBufferView.handle, backFrameBufferView.handle);
			create_framebuffer(window, offlineRenderPass.handle,
				temp_framebuffer, depthBufferView.handle, backBufferView.handle);
			create_framebuffer(window, offlineRenderPass.handle,
				temp_framebuffer_horizontal, depthBufferView.handle, backBufferViewHorizontal.handle);
			create_framebuffer(window, offlineRenderPass.handle,
				temp_framebuffer_vertical, depthBufferView.handle, backBufferViewVertical.handle);

			framebuffers.clear();
			create_swapchain_framebuffers(window, renderPass.handle, framebuffers, depthBufferView.handle);

			updateBackBufferDescriptorSet(window, backFrameBufferDescriptor, backFrameBufferView.handle, filterSampler.handle);
			updateBackBufferDescriptorSet(window, backBufferDescriptor, backBufferView.handle, filterSampler.handle);
			updateBackBufferDescriptorSet(window, backBufferBrightHorizontal, backBufferViewHorizontal.handle,
				filterSampler.handle);
			updateBackBufferDescriptorSet(window, backBufferBrightVertical, backBufferViewVertical.handle,
				filterSampler.handle);

			if (changes.changedSize)
			{
				//pipe = create_pipeline(window, renderPass.handle, pipeLayout.handle);
				pipe = create_pipeline(window, offlineRenderPass.handle, pipeLayout.handle);
				pipe_filter_bright = create_pipeline_filter_bright(window, offlineRenderPass.handle, pipeLayout.handle);
				postPipe = create_postprocess_pipeline(window, renderPass.handle, postPipeLayout.handle);
				filterHorizontalPipe = create_pipeline_horizontal(window, renderPass.handle, postPipeLayout.handle);
				filterVerticalPipe = create_pipeline_vertical(window, renderPass.handle, postPipeLayout.handle);
			}

			recreateSwapchain = false;
			continue;
		}

		std::uint32_t imageIndex = 0;
		auto const acquireRes = vkAcquireNextImageKHR(
			window.device,
			window.swapchain,
			std::numeric_limits<std::uint64_t>::max(),
			imageAvailable.handle,
			VK_NULL_HANDLE,
			&imageIndex
		);

		if (VK_SUBOPTIMAL_KHR == acquireRes ||
			VK_ERROR_OUT_OF_DATE_KHR == acquireRes)
		{
			// This occurs when the window has been resized
			recreateSwapchain = true;
			continue;
		}

		if (VK_SUCCESS != acquireRes)
		{
			throw lut::Error("Unable to acquire enxt swapchain image\n"
				"vkAcquireNextImageKHR() returned() %s", lut::to_string(acquireRes).c_str());
		}

		//TODO: wait for command buffer to be available
		// Make sure command buffer is not in use
		assert(std::size_t(imageIndex) < cbfences.size());

		if (auto const res = vkWaitForFences(window.device, 1,
			&cbfences[imageIndex].handle, VK_TRUE,
			std::numeric_limits<std::uint64_t>::max()); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to wait for command buffer fence %u\n"
				"vkWaitForFences() returned %s", lut::to_string(res).c_str());
		}

		if (auto const res = vkResetFences(window.device, 1,
			&cbfences[imageIndex].handle); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to reset command buffer fence %u\n"
				"vkResetFences() returned %s", lut::to_string(res).c_str());
		}


		//TODO: record and submit commands
		// Record and submit commands for this frame
		assert(std::size_t(imageIndex) < cbuffers.size());
		assert(std::size_t(imageIndex) < framebuffers.size());

		glsl::SceneUniform sceneUniforms{};
		update_scene_uniforms(sceneUniforms, window.swapchainExtent.width, window.swapchainExtent.height, numLight);
		std::vector<glsl::MaterialUniform> materialUniforms(materialDescriptors.size());
		std::vector<glsl::MaterialPBRUniform> materialPBRUniforms(materialPBRDescriptors.size());
		for (size_t i = 0; i < materialUniforms.size(); i++)
		{
			update_material_uniforms(
				materialUniforms[i],
				glm::vec4(carModel.materials[i].emissive, 1),
				glm::vec4(carModel.materials[i].diffuse, 1),
				glm::vec4(carModel.materials[i].specular, 1),
				carModel.materials[i].shininess
			);

			update_material_PBR_uniforms(
				materialPBRUniforms[i],
				glm::vec4(carModel.materials[i].emissive, 1),
				glm::vec4(carModel.materials[i].albedo, 1),
				carModel.materials[i].shininess,
				carModel.materials[i].metalness,
				numLight
			);
		}

		record_commands(
			cbuffers[imageIndex],
			offlineRenderPass.handle,
			renderPass.handle,
			backFramebuffer.handle,
			temp_framebuffer.handle,
			temp_framebuffer_horizontal.handle,
			temp_framebuffer_vertical.handle,
			framebuffers[imageIndex].handle,
			pipe.handle,
			pipe_filter_bright.handle,
			filterHorizontalPipe.handle,
			filterVerticalPipe.handle,
			postPipe.handle,
			window.swapchainExtent,
			loadedModel,
			sceneUBO.buffer,
			sceneUniforms,
			pipeLayout.handle,
			postPipeLayout.handle,
			sceneDescriptor,
			backFrameBufferDescriptor,
			backBufferDescriptor,
			backBufferBrightHorizontal,
			backBufferBrightVertical,
			materialUBO,
			materialUniforms,
			materialDescriptors,
			materialPBRUBO,
			materialPBRUniforms,
			materialPBRDescriptors
		);

		submit_commands(
			window,
			cbuffers[imageIndex],
			cbfences[imageIndex].handle,
			imageAvailable.handle,
			renderFinished.handle
		);

		//TODO: present rendered images.
		// Present the result
		VkPresentInfoKHR presentInfo{};
		presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
		presentInfo.waitSemaphoreCount = 1;
		presentInfo.pWaitSemaphores = &renderFinished.handle;
		presentInfo.swapchainCount = 1;
		presentInfo.pSwapchains = &window.swapchain;
		presentInfo.pImageIndices = &imageIndex;
		presentInfo.pResults = nullptr;

		auto const presentRes = vkQueuePresentKHR(window.presentQueue, &presentInfo);

		if (VK_SUBOPTIMAL_KHR == presentRes || VK_ERROR_OUT_OF_DATE_KHR == presentRes)
		{
			recreateSwapchain = true;
		}
		else if (VK_SUCCESS != presentRes)
		{
			throw lut::Error("Unable present swapchain image%u\n"
				"vkQueuePresentKHR() returned %s", imageIndex,
				lut::to_string(presentRes).c_str());
		}
	}

	vkDeviceWaitIdle(window.device);

	return 0;
}
catch( std::exception const& eErr )
{
	std::fprintf( stderr, "\n" );
	std::fprintf( stderr, "Error: %s\n", eErr.what() );
	return 1;
}

namespace
{
	void glfw_callback_key_press(GLFWwindow* aWindow, int aKey, int aScanCode, int aAction, int aModifierFlags)
	{
		if (GLFW_KEY_ESCAPE == aKey && GLFW_PRESS == aAction)
		{
			glfwSetWindowShouldClose(aWindow, GLFW_TRUE);
		}

		if (GLFW_KEY_LEFT_SHIFT == aKey && (GLFW_REPEAT == aAction || GLFW_PRESS == aAction))
		{
			multiplier = 20;
		}
		if (GLFW_KEY_LEFT_SHIFT == aKey && GLFW_RELEASE == aAction)
		{
			multiplier = 5;
		}

		if (GLFW_KEY_LEFT_CONTROL == aKey && (GLFW_REPEAT == aAction || GLFW_PRESS == aAction))
		{
			multiplier = 1;
		}
		if (GLFW_KEY_LEFT_CONTROL == aKey && GLFW_RELEASE == aAction)
		{
			multiplier = 2;
		}

		if (GLFW_KEY_W == aKey && (GLFW_REPEAT == aAction || GLFW_PRESS == aAction))
		{
			position.z = position.z + 0.01 * multiplier;
		}
		if (GLFW_KEY_S == aKey && (GLFW_REPEAT == aAction || GLFW_PRESS == aAction))
		{
			position.z = position.z - 0.01 * multiplier;
		}
		if (GLFW_KEY_A == aKey && (GLFW_REPEAT == aAction || GLFW_PRESS == aAction))
		{
			position.x = position.x + 0.01 * multiplier;
		}
		if (GLFW_KEY_D == aKey && (GLFW_REPEAT == aAction || GLFW_PRESS == aAction))
		{
			position.x = position.x - 0.01 * multiplier;
		}
		if (GLFW_KEY_E == aKey && (GLFW_REPEAT == aAction || GLFW_PRESS == aAction))
		{
			position.y = position.y - 0.01 * multiplier;
		}
		if (GLFW_KEY_Q == aKey && (GLFW_REPEAT == aAction || GLFW_PRESS == aAction))
		{
			position.y = position.y + 0.01 * multiplier;
		}

		// Lights
		if (GLFW_KEY_1 == aKey && (GLFW_REPEAT == aAction || GLFW_PRESS == aAction))
		{
			numLight = 1;
		}
		if (GLFW_KEY_2 == aKey && (GLFW_REPEAT == aAction || GLFW_PRESS == aAction))
		{
			numLight = 2;
		}
		if (GLFW_KEY_3 == aKey && (GLFW_REPEAT == aAction || GLFW_PRESS == aAction))
		{
			numLight = 3;
		}
	}

	void glfw_callback_mouse_press(GLFWwindow* aWindow, int button, int aAction, int aModifierFlags)
	{
		if (GLFW_MOUSE_BUTTON_2 == button && GLFW_PRESS == aAction)
		{
			moveCamera = !moveCamera;
		}
	}

	void glfw_callback_mouse_position(GLFWwindow* aWindow, double xpos, double ypos)
	{
		if (moveCamera)
		{
			double lastPosX = mouseX;
			double lastPosY = mouseY;

			double movementX = xpos - lastPosX;
			double movementY = ypos - lastPosY;

			rotation.x = rotation.x + movementX;
			rotation.y = rotation.y + movementY;
		}

		mouseX = xpos;
		mouseY = ypos;
	}

	void update_scene_uniforms(glsl::SceneUniform& aSceneUniforms, std::uint32_t aFramebufferWidth, 
		std::uint32_t aFramebufferHeight, int numLight)
	{
		float const aspect = aFramebufferWidth / float(aFramebufferHeight);

		aSceneUniforms.projection = glm::perspectiveRH_ZO(
			lut::Radians(cfg::kCameraFov).value(),
			aspect,
			cfg::kCameraNear,
			cfg::kCameraFar
		);

		aSceneUniforms.projection[1][1] *= -1.0f; // mirror Y axis
		glm::mat4 rotationMatrix = glm::eulerAngleXYZ(rotation.y * 0.005, rotation.x * 0.005, rotation.z * 0.005);
		aSceneUniforms.camera =  glm::translate(position) * rotationMatrix;
		aSceneUniforms.projcam = aSceneUniforms.projection * aSceneUniforms.camera;

		aSceneUniforms.cameraPos = glm::vec4(0, 0, 0, 0);

		//aSceneUniforms.lightPos[0] = glm::vec4(-20, 15.3f, -3.0f, 1.0f);
		//aSceneUniforms.lightColor[0] = glm::vec4(1.0f, 0.0f, 0.0f, 1.0f);
		//aSceneUniforms.lightPos[1] = glm::vec4(0, 15.3f, -3.0f, 1.0f);
		//aSceneUniforms.lightColor[1] = glm::vec4(0.0f, 1.0f, 0.0f, 1.0f);
		//aSceneUniforms.lightPos[2] = glm::vec4(20, 15.3f, -3.0f, 1.0f);
		//aSceneUniforms.lightColor[2] = glm::vec4(0.0f, 0.0f, 1.0f, 1.0f);

		aSceneUniforms.rotation = rotationMatrix;

		aSceneUniforms.size = numLight;

		// Default
		aSceneUniforms.lightPos[0] = glm::vec4(0, 9.3f, -3.0f, 1.0f);
		aSceneUniforms.lightColor[0] = glm::vec4(1.0f, 1.0f, 0.8f, 1.0f);
		//aSceneUniforms.lightColor[0] = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
		aSceneUniforms.lightPos[1] = glm::vec4(0, 9.3f, -3.0f, 1.0f);
		aSceneUniforms.lightColor[1] = glm::vec4(0.0f, 1.0f, 0.0f, 1.0f);
		aSceneUniforms.lightPos[2] = glm::vec4(0, 9.3f, -3.0f, 1.0f);
		aSceneUniforms.lightColor[2] = glm::vec4(0.0f, 0.0f, 1.0f, 1.0f);
	}

	void update_material_uniforms(glsl::MaterialUniform& aMaterialUniforms, glm::vec4 emissive,
		glm::vec4 diffuse, glm::vec4 specular, float shininess)
	{
		aMaterialUniforms.emissive = emissive;
		aMaterialUniforms.diffuse = diffuse;
		aMaterialUniforms.specular = specular;
		aMaterialUniforms.shininess = shininess;
	}

	void update_material_PBR_uniforms(glsl::MaterialPBRUniform& aMaterialUniforms, glm::vec4 emissive,
		glm::vec4 albedo, float shininess, float metalness, int numLight)
	{
		aMaterialUniforms.emissive = emissive;
		aMaterialUniforms.albedo = albedo;
		aMaterialUniforms.shininess = shininess;
		aMaterialUniforms.metalness = metalness;
		aMaterialUniforms.size = numLight;
	}

	lut::RenderPass create_render_pass(lut::VulkanWindow const& aWindow)
	{
		VkAttachmentDescription attachments[2]{};
		attachments[0].format = aWindow.swapchainFormat;
		attachments[0].samples = VK_SAMPLE_COUNT_1_BIT;
		attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		attachments[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		attachments[0].finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

		attachments[1].format = cfg::kDepthFormat;
		attachments[1].samples = VK_SAMPLE_COUNT_1_BIT;
		attachments[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		attachments[1].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		attachments[1].initialLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
		attachments[1].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

		VkAttachmentReference subpassAttachments[1]{};
		subpassAttachments[0].attachment = 0;
		subpassAttachments[0].layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		VkAttachmentReference depthAttachments{};
		depthAttachments.attachment = 1;
		depthAttachments.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

		VkSubpassDescription subpasses[1]{};
		subpasses[0].pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		subpasses[0].colorAttachmentCount = sizeof(subpassAttachments) / sizeof(subpassAttachments[0]);
		subpasses[0].pColorAttachments = subpassAttachments;
		subpasses[0].pDepthStencilAttachment = &depthAttachments;

		VkRenderPassCreateInfo passInfo{};
		passInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
		passInfo.attachmentCount = 2;
		passInfo.pAttachments = attachments;
		passInfo.subpassCount = 1;
		passInfo.pSubpasses = subpasses;
		passInfo.dependencyCount = 0;
		//passInfo.pDependencies = dependencies;

		VkRenderPass rpass = VK_NULL_HANDLE;
		if (auto const res = vkCreateRenderPass(aWindow.device, &passInfo,
			nullptr, &rpass); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to create render pass\n"
				"vkCreateRenderPass() returned %s", lut::to_string(res).c_str());
		}

		return lut::RenderPass(aWindow.device, rpass);
	}

	lut::RenderPass create_render_pass_texture(lut::VulkanWindow const& aWindow)
	{
		VkAttachmentDescription attachments[2]{};
		//attachments[0].format = VK_FORMAT_R8G8B8A8_SRGB;
		attachments[0].format = aWindow.swapchainFormat;;
		attachments[0].samples = VK_SAMPLE_COUNT_1_BIT;
		attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		attachments[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		attachments[0].finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

		attachments[1].format = cfg::kDepthFormat;
		attachments[1].samples = VK_SAMPLE_COUNT_1_BIT;
		attachments[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		attachments[1].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		attachments[1].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		attachments[1].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

		VkAttachmentReference subpassAttachments[1]{};
		subpassAttachments[0].attachment = 0;
		subpassAttachments[0].layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		VkAttachmentReference depthAttachments{};
		depthAttachments.attachment = 1;
		depthAttachments.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

		VkSubpassDescription subpasses[1]{};
		subpasses[0].pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		subpasses[0].colorAttachmentCount = 1;
		subpasses[0].pColorAttachments = subpassAttachments;
		subpasses[0].pDepthStencilAttachment = &depthAttachments;

		VkSubpassDependency dependencies[2]{};
		dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
		dependencies[0].dstSubpass = 0;
		dependencies[0].srcStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
		dependencies[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		dependencies[0].srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
		dependencies[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
		dependencies[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

		dependencies[1].srcSubpass = 0;
		dependencies[1].dstSubpass = VK_SUBPASS_EXTERNAL;
		dependencies[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		dependencies[1].dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
		dependencies[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
		dependencies[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
		dependencies[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;
		
		VkRenderPassCreateInfo passInfo{};
		passInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
		passInfo.attachmentCount = sizeof(attachments) / sizeof(attachments[0]);
		passInfo.pAttachments = attachments;
		passInfo.subpassCount = 1;
		passInfo.pSubpasses = subpasses;
		passInfo.dependencyCount = 2;
		passInfo.pDependencies = dependencies;

		VkRenderPass rpass = VK_NULL_HANDLE;
		if (auto const res = vkCreateRenderPass(aWindow.device, &passInfo,
			nullptr, &rpass); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to create render pass for texture\n"
				"vkCreateRenderPass() returned %s", lut::to_string(res).c_str());
		}

		return lut::RenderPass(aWindow.device, rpass);
	}

	lut::DescriptorSetLayout create_scene_descriptor_layout(lut::VulkanWindow const& aWindow)
	{
		VkDescriptorSetLayoutBinding bindings[1]{};
		bindings[0].binding = 0;
		bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		bindings[0].descriptorCount = 1;
		bindings[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

		VkDescriptorSetLayoutCreateInfo layoutInfo{};
		layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		layoutInfo.bindingCount = sizeof(bindings) / sizeof(bindings[0]);
		layoutInfo.pBindings = bindings;

		VkDescriptorSetLayout layout = VK_NULL_HANDLE;
		if (auto const res = vkCreateDescriptorSetLayout(aWindow.device, &layoutInfo, nullptr, &layout);
			VK_SUCCESS != res)
		{
			throw lut::Error("Unable to create descriptor set layout\n"
				"vkCreateDescriptorSetLayout() returned %s", lut::to_string(res).c_str());
		}

		return lut::DescriptorSetLayout(aWindow.device, layout);
	}

	lut::DescriptorSetLayout create_material_descriptor_layout(lut::VulkanWindow const& aWindow)
	{
		VkDescriptorSetLayoutBinding bindings[1]{};
		bindings[0].binding = 0;
		bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		bindings[0].descriptorCount = 1;
		bindings[0].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

		VkDescriptorSetLayoutCreateInfo layoutInfo{};
		layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		layoutInfo.bindingCount = sizeof(bindings) / sizeof(bindings[0]);
		layoutInfo.pBindings = bindings;

		VkDescriptorSetLayout layout = VK_NULL_HANDLE;
		if (auto const res = vkCreateDescriptorSetLayout(aWindow.device, &layoutInfo, nullptr, &layout);
			VK_SUCCESS != res)
		{
			throw lut::Error("Unable to create descriptor set layout\n"
				"vkCreateDescriptorSetLayout() returned %s", lut::to_string(res).c_str());
		}

		return lut::DescriptorSetLayout(aWindow.device, layout);
	}

	lut::DescriptorSetLayout create_object_descriptor_layout(lut::VulkanWindow const& aWindow)
	{
		VkDescriptorSetLayoutBinding bindings[1]{};
		bindings[0].binding = 0;
		bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		bindings[0].descriptorCount = 1;
		bindings[0].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

		VkDescriptorSetLayoutCreateInfo layoutInfo{};
		layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		layoutInfo.bindingCount = sizeof(bindings) / sizeof(bindings[0]);
		layoutInfo.pBindings = bindings;

		VkDescriptorSetLayout layout = VK_NULL_HANDLE;
		if (auto const res = vkCreateDescriptorSetLayout(aWindow.device, &layoutInfo, nullptr, &layout);
			VK_SUCCESS != res)
		{
			throw lut::Error("Unable to create descriptor set layout\n"
				"vkCreateDescriptorSetLayout() returned %s", lut::to_string(res).c_str());
		}

		return lut::DescriptorSetLayout(aWindow.device, layout);
	}

	lut::PipelineLayout create_pipeline_layout(lut::VulkanContext const& aContext, 
		VkDescriptorSetLayout aSceneLayout, VkDescriptorSetLayout aMaterialLayout, 
		VkDescriptorSetLayout aObjectLayout)
	{
		VkDescriptorSetLayout layouts[]
		{
			aSceneLayout, // set 0
			aMaterialLayout, // set 1
			aMaterialLayout, // set 2
			aObjectLayout,  // set 3
		};

		VkPipelineLayoutCreateInfo layoutInfo{};
		layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		layoutInfo.setLayoutCount = sizeof(layouts) / sizeof(layouts[0]);
		layoutInfo.pSetLayouts = layouts;
		layoutInfo.pushConstantRangeCount = 0;
		layoutInfo.pPushConstantRanges = nullptr;

		VkPipelineLayout layout = VK_NULL_HANDLE;
		if (auto const res = vkCreatePipelineLayout(aContext.device,
			&layoutInfo, nullptr, &layout); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to create pipeline layout\n"
				"vkCreatePipelineLayout() returned %s", lut::to_string(res).c_str());
		}

		return lut::PipelineLayout(aContext.device, layout);
	}

	lut::PipelineLayout create_pipeline_with_texture_layout(lut::VulkanContext const& aContext, 
		VkDescriptorSetLayout aSceneLayout, VkDescriptorSetLayout aObjectLayout)
	{
		VkDescriptorSetLayout layouts[]
		{
			aSceneLayout, // set 0
			aObjectLayout // set 1
		};

		VkPipelineLayoutCreateInfo layoutInfo{};
		layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		layoutInfo.setLayoutCount = sizeof(layouts) / sizeof(layouts[0]);
		layoutInfo.pSetLayouts = layouts;
		layoutInfo.pushConstantRangeCount = 0;
		layoutInfo.pPushConstantRanges = nullptr;

		VkPipelineLayout layout = VK_NULL_HANDLE;
		if (auto const res = vkCreatePipelineLayout(aContext.device,
			&layoutInfo, nullptr, &layout); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to create pipeline layout\n"
				"vkCreatePipelineLayout() returned %s", lut::to_string(res).c_str());
		}

		return lut::PipelineLayout(aContext.device, layout);
	}

	lut::PipelineLayout create_postprocess_pipeline_layout(lut::VulkanContext const& aContext,
		VkDescriptorSetLayout aSceneLayout, VkDescriptorSetLayout aObjectLayout)
	{
		VkDescriptorSetLayout layouts[]
		{
			aObjectLayout, // set 0
			aObjectLayout, // set 1
		};

		VkPipelineLayoutCreateInfo layoutInfo{};
		layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		layoutInfo.setLayoutCount = sizeof(layouts) / sizeof(layouts[0]);
		layoutInfo.pSetLayouts = layouts;
		layoutInfo.pushConstantRangeCount = 0;
		layoutInfo.pPushConstantRanges = nullptr;

		VkPipelineLayout layout = VK_NULL_HANDLE;
		if (auto const res = vkCreatePipelineLayout(aContext.device,
			&layoutInfo, nullptr, &layout); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to create pipeline layout\n"
				"vkCreatePipelineLayout() returned %s", lut::to_string(res).c_str());
		}

		return lut::PipelineLayout(aContext.device, layout);
	}

	lut::Pipeline create_pipeline_horizontal(lut::VulkanWindow const& aWindow, VkRenderPass aRenderPass, VkPipelineLayout aPipelineLayout)
	{
		// Load shader modules
		lut::ShaderModule vert = lut::load_shader_module(aWindow, cfg::kHorizontalFilterVertPath);
		lut::ShaderModule frag = lut::load_shader_module(aWindow, cfg::kHorizontalFilterFragPath);

		// Define shader stages in the pipeline
		// Two stages, 1. Vertex shader 2. Fragment shader
		VkPipelineShaderStageCreateInfo stages[2]{};
		stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
		stages[0].module = vert.handle;
		stages[0].pName = "main";

		stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		stages[1].module = frag.handle;
		stages[1].pName = "main";

		VkPipelineVertexInputStateCreateInfo inputInfo{};
		inputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
		inputInfo.vertexBindingDescriptionCount = 0;
		inputInfo.pVertexBindingDescriptions = nullptr;
		inputInfo.vertexAttributeDescriptionCount = 0;
		inputInfo.pVertexAttributeDescriptions = nullptr;

		// Define which primitive (point, line, triangle,...)
		VkPipelineInputAssemblyStateCreateInfo assemblyInfo{};
		assemblyInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
		assemblyInfo.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
		assemblyInfo.primitiveRestartEnable = VK_FALSE;

		// Define viewport and scissor regions
		VkViewport viewport{};
		viewport.x = 0.0f;
		viewport.y = 0.0f;
		viewport.width = float(aWindow.swapchainExtent.width);
		viewport.height = float(aWindow.swapchainExtent.height);
		viewport.minDepth = 0.0f;
		viewport.maxDepth = 1.0f;

		VkRect2D scissor{};
		scissor.offset = VkOffset2D{ 0, 0 };
		scissor.extent = VkExtent2D{ aWindow.swapchainExtent.width,
			aWindow.swapchainExtent.height };

		VkPipelineViewportStateCreateInfo viewportInfo{};
		viewportInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
		viewportInfo.viewportCount = 1;
		viewportInfo.pViewports = &viewport;
		viewportInfo.scissorCount = 1;
		viewportInfo.pScissors = &scissor;

		// Define rasterization options
		VkPipelineRasterizationStateCreateInfo rasterInfo{};
		rasterInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
		rasterInfo.depthClampEnable = VK_FALSE;
		rasterInfo.rasterizerDiscardEnable = VK_FALSE;
		rasterInfo.polygonMode = VK_POLYGON_MODE_FILL;
		rasterInfo.cullMode = VK_CULL_MODE_FRONT_BIT;
		rasterInfo.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
		rasterInfo.depthBiasEnable = VK_FALSE;
		rasterInfo.lineWidth = 1.0f;

		// Define multisampling state
		VkPipelineMultisampleStateCreateInfo samplingInfo{};
		samplingInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
		samplingInfo.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

		// Define blend state
		// i.e. which color channels to write
		VkPipelineColorBlendAttachmentState blendStates[1]{};
		blendStates[0].blendEnable = VK_FALSE;
		blendStates[0].colorBlendOp = VK_BLEND_OP_ADD;
		blendStates[0].srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
		blendStates[0].dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
		blendStates[0].colorWriteMask = VK_COLOR_COMPONENT_R_BIT |
			VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT |
			VK_COLOR_COMPONENT_A_BIT;

		VkPipelineColorBlendStateCreateInfo blendInfo{};
		blendInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
		blendInfo.logicOpEnable = VK_FALSE;
		blendInfo.attachmentCount = 1;
		blendInfo.pAttachments = blendStates;

		// Depth Testing
		VkPipelineDepthStencilStateCreateInfo depthInfo{};
		depthInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
		depthInfo.depthTestEnable = VK_TRUE;
		depthInfo.depthWriteEnable = VK_TRUE;
		depthInfo.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
		depthInfo.minDepthBounds = 0.0f;
		depthInfo.maxDepthBounds = 1.0f;

		// Create pipeline
		VkGraphicsPipelineCreateInfo pipeInfo{};
		pipeInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;

		pipeInfo.stageCount = 2; // Vertex and fragment stages
		pipeInfo.pStages = stages;

		pipeInfo.pVertexInputState = &inputInfo;
		pipeInfo.pInputAssemblyState = &assemblyInfo;
		pipeInfo.pTessellationState = nullptr;
		pipeInfo.pViewportState = &viewportInfo;
		pipeInfo.pRasterizationState = &rasterInfo;
		pipeInfo.pMultisampleState = &samplingInfo;
		pipeInfo.pDepthStencilState = &depthInfo;
		pipeInfo.pColorBlendState = &blendInfo;
		pipeInfo.pDynamicState = nullptr;
		pipeInfo.layout = aPipelineLayout;
		pipeInfo.renderPass = aRenderPass;
		pipeInfo.subpass = 0;

		VkPipeline pipe = VK_NULL_HANDLE;
		if (auto const res = vkCreateGraphicsPipelines(aWindow.device,
			VK_NULL_HANDLE, 1, &pipeInfo, nullptr, &pipe); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to create graphics pipeline\n"
				"vkCreateGraphicsPipelines() returned %s", lut::to_string(res).c_str());
		}

		return lut::Pipeline(aWindow.device, pipe);
	}

	lut::Pipeline create_pipeline_vertical(lut::VulkanWindow const& aWindow, VkRenderPass aRenderPass, VkPipelineLayout aPipelineLayout)
	{
		// Load shader modules
		lut::ShaderModule vert = lut::load_shader_module(aWindow, cfg::kVerticalFilterVertPath);
		lut::ShaderModule frag = lut::load_shader_module(aWindow, cfg::kVerticalFilterFragPath);

		// Define shader stages in the pipeline
		// Two stages, 1. Vertex shader 2. Fragment shader
		VkPipelineShaderStageCreateInfo stages[2]{};
		stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
		stages[0].module = vert.handle;
		stages[0].pName = "main";

		stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		stages[1].module = frag.handle;
		stages[1].pName = "main";

		VkPipelineVertexInputStateCreateInfo inputInfo{};
		inputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
		inputInfo.vertexBindingDescriptionCount = 0;
		inputInfo.pVertexBindingDescriptions = nullptr;
		inputInfo.vertexAttributeDescriptionCount = 0;
		inputInfo.pVertexAttributeDescriptions = nullptr;

		// Define which primitive (point, line, triangle,...)
		VkPipelineInputAssemblyStateCreateInfo assemblyInfo{};
		assemblyInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
		assemblyInfo.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
		assemblyInfo.primitiveRestartEnable = VK_FALSE;

		// Define viewport and scissor regions
		VkViewport viewport{};
		viewport.x = 0.0f;
		viewport.y = 0.0f;
		viewport.width = float(aWindow.swapchainExtent.width);
		viewport.height = float(aWindow.swapchainExtent.height);
		viewport.minDepth = 0.0f;
		viewport.maxDepth = 1.0f;

		VkRect2D scissor{};
		scissor.offset = VkOffset2D{ 0, 0 };
		scissor.extent = VkExtent2D{ aWindow.swapchainExtent.width,
			aWindow.swapchainExtent.height };

		VkPipelineViewportStateCreateInfo viewportInfo{};
		viewportInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
		viewportInfo.viewportCount = 1;
		viewportInfo.pViewports = &viewport;
		viewportInfo.scissorCount = 1;
		viewportInfo.pScissors = &scissor;

		// Define rasterization options
		VkPipelineRasterizationStateCreateInfo rasterInfo{};
		rasterInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
		rasterInfo.depthClampEnable = VK_FALSE;
		rasterInfo.rasterizerDiscardEnable = VK_FALSE;
		rasterInfo.polygonMode = VK_POLYGON_MODE_FILL;
		rasterInfo.cullMode = VK_CULL_MODE_FRONT_BIT;
		rasterInfo.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
		rasterInfo.depthBiasEnable = VK_FALSE;
		rasterInfo.lineWidth = 1.0f;

		// Define multisampling state
		VkPipelineMultisampleStateCreateInfo samplingInfo{};
		samplingInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
		samplingInfo.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

		// Define blend state
		// i.e. which color channels to write
		VkPipelineColorBlendAttachmentState blendStates[1]{};
		blendStates[0].blendEnable = VK_FALSE;
		blendStates[0].colorBlendOp = VK_BLEND_OP_ADD;
		blendStates[0].srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
		blendStates[0].dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
		blendStates[0].colorWriteMask = VK_COLOR_COMPONENT_R_BIT |
			VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT |
			VK_COLOR_COMPONENT_A_BIT;

		VkPipelineColorBlendStateCreateInfo blendInfo{};
		blendInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
		blendInfo.logicOpEnable = VK_FALSE;
		blendInfo.attachmentCount = 1;
		blendInfo.pAttachments = blendStates;

		// Depth Testing
		VkPipelineDepthStencilStateCreateInfo depthInfo{};
		depthInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
		depthInfo.depthTestEnable = VK_TRUE;
		depthInfo.depthWriteEnable = VK_TRUE;
		depthInfo.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
		depthInfo.minDepthBounds = 0.0f;
		depthInfo.maxDepthBounds = 1.0f;

		// Create pipeline
		VkGraphicsPipelineCreateInfo pipeInfo{};
		pipeInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;

		pipeInfo.stageCount = 2; // Vertex and fragment stages
		pipeInfo.pStages = stages;

		pipeInfo.pVertexInputState = &inputInfo;
		pipeInfo.pInputAssemblyState = &assemblyInfo;
		pipeInfo.pTessellationState = nullptr;
		pipeInfo.pViewportState = &viewportInfo;
		pipeInfo.pRasterizationState = &rasterInfo;
		pipeInfo.pMultisampleState = &samplingInfo;
		pipeInfo.pDepthStencilState = &depthInfo;
		pipeInfo.pColorBlendState = &blendInfo;
		pipeInfo.pDynamicState = nullptr;
		pipeInfo.layout = aPipelineLayout;
		pipeInfo.renderPass = aRenderPass;
		pipeInfo.subpass = 0;

		VkPipeline pipe = VK_NULL_HANDLE;
		if (auto const res = vkCreateGraphicsPipelines(aWindow.device,
			VK_NULL_HANDLE, 1, &pipeInfo, nullptr, &pipe); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to create graphics pipeline\n"
				"vkCreateGraphicsPipelines() returned %s", lut::to_string(res).c_str());
		}

		return lut::Pipeline(aWindow.device, pipe);
	}

	lut::Pipeline create_postprocess_pipeline(lut::VulkanWindow const& aWindow, VkRenderPass aRenderPass, VkPipelineLayout aPipelineLayout)
	{
		// Load shader modules
		lut::ShaderModule vert = lut::load_shader_module(aWindow, cfg::kPostProcessinVertgPath);
		lut::ShaderModule frag = lut::load_shader_module(aWindow, cfg::kPostProcessingFragPath);

		// Define shader stages in the pipeline
		// Two stages, 1. Vertex shader 2. Fragment shader
		VkPipelineShaderStageCreateInfo stages[2]{};
		stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
		stages[0].module = vert.handle;
		stages[0].pName = "main";

		stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		stages[1].module = frag.handle;
		stages[1].pName = "main";

		VkPipelineVertexInputStateCreateInfo inputInfo{};
		inputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
		inputInfo.vertexBindingDescriptionCount = 0;
		inputInfo.pVertexBindingDescriptions = nullptr;
		inputInfo.vertexAttributeDescriptionCount = 0;
		inputInfo.pVertexAttributeDescriptions = nullptr;

		// Define which primitive (point, line, triangle,...)
		VkPipelineInputAssemblyStateCreateInfo assemblyInfo{};
		assemblyInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
		assemblyInfo.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
		assemblyInfo.primitiveRestartEnable = VK_FALSE;

		// Define viewport and scissor regions
		VkViewport viewport{};
		viewport.x = 0.0f;
		viewport.y = 0.0f;
		viewport.width = float(aWindow.swapchainExtent.width);
		viewport.height = float(aWindow.swapchainExtent.height);
		viewport.minDepth = 0.0f;
		viewport.maxDepth = 1.0f;

		VkRect2D scissor{};
		scissor.offset = VkOffset2D{ 0, 0 };
		scissor.extent = VkExtent2D{ aWindow.swapchainExtent.width,
			aWindow.swapchainExtent.height };

		VkPipelineViewportStateCreateInfo viewportInfo{};
		viewportInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
		viewportInfo.viewportCount = 1;
		viewportInfo.pViewports = &viewport;
		viewportInfo.scissorCount = 1;
		viewportInfo.pScissors = &scissor;

		// Define rasterization options
		VkPipelineRasterizationStateCreateInfo rasterInfo{};
		rasterInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
		rasterInfo.depthClampEnable = VK_FALSE;
		rasterInfo.rasterizerDiscardEnable = VK_FALSE;
		rasterInfo.polygonMode = VK_POLYGON_MODE_FILL;
		rasterInfo.cullMode = VK_CULL_MODE_FRONT_BIT;
		rasterInfo.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
		rasterInfo.depthBiasEnable = VK_FALSE;
		rasterInfo.lineWidth = 1.0f;

		// Define multisampling state
		VkPipelineMultisampleStateCreateInfo samplingInfo{};
		samplingInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
		samplingInfo.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

		// Define blend state
		// i.e. which color channels to write
		VkPipelineColorBlendAttachmentState blendStates[1]{};
		blendStates[0].blendEnable = VK_FALSE;
		blendStates[0].colorBlendOp = VK_BLEND_OP_ADD;
		blendStates[0].srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
		blendStates[0].dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
		blendStates[0].colorWriteMask = VK_COLOR_COMPONENT_R_BIT |
			VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT |
			VK_COLOR_COMPONENT_A_BIT;

		VkPipelineColorBlendStateCreateInfo blendInfo{};
		blendInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
		blendInfo.logicOpEnable = VK_FALSE;
		blendInfo.attachmentCount = 1;
		blendInfo.pAttachments = blendStates;

		// Depth Testing
		VkPipelineDepthStencilStateCreateInfo depthInfo{};
		depthInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
		depthInfo.depthTestEnable = VK_TRUE;
		depthInfo.depthWriteEnable = VK_TRUE;
		depthInfo.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
		depthInfo.minDepthBounds = 0.0f;
		depthInfo.maxDepthBounds = 1.0f;

		// Create pipeline
		VkGraphicsPipelineCreateInfo pipeInfo{};
		pipeInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;

		pipeInfo.stageCount = 2; // Vertex and fragment stages
		pipeInfo.pStages = stages;

		pipeInfo.pVertexInputState = &inputInfo;
		pipeInfo.pInputAssemblyState = &assemblyInfo;
		pipeInfo.pTessellationState = nullptr;
		pipeInfo.pViewportState = &viewportInfo;
		pipeInfo.pRasterizationState = &rasterInfo;
		pipeInfo.pMultisampleState = &samplingInfo;
		pipeInfo.pDepthStencilState = &depthInfo;
		pipeInfo.pColorBlendState = &blendInfo;
		pipeInfo.pDynamicState = nullptr;
		pipeInfo.layout = aPipelineLayout;
		pipeInfo.renderPass = aRenderPass;
		pipeInfo.subpass = 0;

		VkPipeline pipe = VK_NULL_HANDLE;
		if (auto const res = vkCreateGraphicsPipelines(aWindow.device,
			VK_NULL_HANDLE, 1, &pipeInfo, nullptr, &pipe); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to create graphics pipeline\n"
				"vkCreateGraphicsPipelines() returned %s", lut::to_string(res).c_str());
		}

		return lut::Pipeline(aWindow.device, pipe);
	}

	lut::Pipeline create_pipeline_with_texture(lut::VulkanWindow const& aWindow, VkRenderPass aRenderPass, VkPipelineLayout aPipelineLayout)
	{
		// Load shader modules
		lut::ShaderModule vert = lut::load_shader_module(aWindow, cfg::kVertShaderPath);
		lut::ShaderModule frag = lut::load_shader_module(aWindow, cfg::kFragTexShaderPath);

		// Define shader stages in the pipeline
		// Two stages, 1. Vertex shader 2. Fragment shader
		VkPipelineShaderStageCreateInfo stages[2]{};
		stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
		stages[0].module = vert.handle;
		stages[0].pName = "main";

		stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		stages[1].module = frag.handle;
		stages[1].pName = "main";

		VkVertexInputBindingDescription vertexInputs[5]{};
		// position
		vertexInputs[0].binding = 0;
		vertexInputs[0].stride = sizeof(glm::vec3);
		vertexInputs[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
		// normal
		vertexInputs[1].binding = 1;
		vertexInputs[1].stride = sizeof(glm::vec3);
		vertexInputs[1].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
		// texcoords
		vertexInputs[2].binding = 2;
		vertexInputs[2].stride = sizeof(glm::vec2);
		vertexInputs[2].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
		// colours
		vertexInputs[3].binding = 3;
		vertexInputs[3].stride = sizeof(glm::vec3);
		vertexInputs[3].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
		// surface normals
		vertexInputs[4].binding = 4;
		vertexInputs[4].stride = sizeof(glm::vec3);
		vertexInputs[4].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

		VkVertexInputAttributeDescription vertexAttributes[5]{};
		// position
		vertexAttributes[0].binding = 0;
		vertexAttributes[0].location = 0;
		vertexAttributes[0].format = VK_FORMAT_R32G32B32_SFLOAT;
		vertexAttributes[0].offset = 0;
		// normal
		vertexAttributes[1].binding = 1;
		vertexAttributes[1].location = 1;
		vertexAttributes[1].format = VK_FORMAT_R32G32B32_SFLOAT;
		vertexAttributes[1].offset = 0;
		// texcoords
		vertexAttributes[2].binding = 2;
		vertexAttributes[2].location = 2;
		vertexAttributes[2].format = VK_FORMAT_R32G32_SFLOAT;
		vertexAttributes[2].offset = 0;
		// colours
		vertexAttributes[3].binding = 3;
		vertexAttributes[3].location = 3;
		vertexAttributes[3].format = VK_FORMAT_R32G32B32_SFLOAT;
		vertexAttributes[3].offset = 0;
		// colours
		vertexAttributes[4].binding = 4;
		vertexAttributes[4].location = 4;
		vertexAttributes[4].format = VK_FORMAT_R32G32B32_SFLOAT;
		vertexAttributes[4].offset = 0;

		VkPipelineVertexInputStateCreateInfo inputInfo{};
		inputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
		inputInfo.vertexBindingDescriptionCount = 4;
		inputInfo.pVertexBindingDescriptions = vertexInputs;
		inputInfo.vertexAttributeDescriptionCount = 4;
		inputInfo.pVertexAttributeDescriptions = vertexAttributes;

		// Define which primitive (point, line, triangle,...)
		VkPipelineInputAssemblyStateCreateInfo assemblyInfo{};
		assemblyInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
		assemblyInfo.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
		assemblyInfo.primitiveRestartEnable = VK_FALSE;

		// Define viewport and scissor regions
		VkViewport viewport{};
		viewport.x = 0.0f;
		viewport.y = 0.0f;
		viewport.width = float(aWindow.swapchainExtent.width);
		viewport.height = float(aWindow.swapchainExtent.height);
		viewport.minDepth = 0.0f;
		viewport.maxDepth = 1.0f;

		VkRect2D scissor{};
		scissor.offset = VkOffset2D{ 0, 0 };
		scissor.extent = VkExtent2D{ aWindow.swapchainExtent.width,
			aWindow.swapchainExtent.height };

		VkPipelineViewportStateCreateInfo viewportInfo{};
		viewportInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
		viewportInfo.viewportCount = 1;
		viewportInfo.pViewports = &viewport;
		viewportInfo.scissorCount = 1;
		viewportInfo.pScissors = &scissor;

		// Define rasterization options
		VkPipelineRasterizationStateCreateInfo rasterInfo{};
		rasterInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
		rasterInfo.depthClampEnable = VK_FALSE;
		rasterInfo.rasterizerDiscardEnable = VK_FALSE;
		rasterInfo.polygonMode = VK_POLYGON_MODE_FILL;
		rasterInfo.cullMode = VK_CULL_MODE_BACK_BIT;
		rasterInfo.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
		rasterInfo.depthBiasEnable = VK_FALSE;
		rasterInfo.lineWidth = 1.0f;

		// Define multisampling state
		VkPipelineMultisampleStateCreateInfo samplingInfo{};
		samplingInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
		samplingInfo.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

		// Define blend state
		// i.e. which color channels to write
		VkPipelineColorBlendAttachmentState blendStates[1]{};
		blendStates[0].blendEnable = VK_FALSE;
		blendStates[0].colorBlendOp = VK_BLEND_OP_ADD;
		blendStates[0].srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
		blendStates[0].dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
		blendStates[0].colorWriteMask = VK_COLOR_COMPONENT_R_BIT |
			VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT |
			VK_COLOR_COMPONENT_A_BIT;

		VkPipelineColorBlendStateCreateInfo blendInfo{};
		blendInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
		blendInfo.logicOpEnable = VK_FALSE;
		blendInfo.attachmentCount = 1;
		blendInfo.pAttachments = blendStates;

		// Depth Testing
		VkPipelineDepthStencilStateCreateInfo depthInfo{};
		depthInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
		depthInfo.depthTestEnable = VK_TRUE;
		depthInfo.depthWriteEnable = VK_TRUE;
		depthInfo.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
		depthInfo.minDepthBounds = 0.0f;
		depthInfo.maxDepthBounds = 1.0f;

		// Create pipeline
		VkGraphicsPipelineCreateInfo pipeInfo{};
		pipeInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;

		pipeInfo.stageCount = 2; // Vertex and fragment stages
		pipeInfo.pStages = stages;

		pipeInfo.pVertexInputState = &inputInfo;
		pipeInfo.pInputAssemblyState = &assemblyInfo;
		pipeInfo.pTessellationState = nullptr;
		pipeInfo.pViewportState = &viewportInfo;
		pipeInfo.pRasterizationState = &rasterInfo;
		pipeInfo.pMultisampleState = &samplingInfo;
		pipeInfo.pDepthStencilState = &depthInfo;
		pipeInfo.pColorBlendState = &blendInfo;
		pipeInfo.pDynamicState = nullptr;
		pipeInfo.layout = aPipelineLayout;
		pipeInfo.renderPass = aRenderPass;
		pipeInfo.subpass = 0;

		VkPipeline pipe = VK_NULL_HANDLE;
		if (auto const res = vkCreateGraphicsPipelines(aWindow.device,
			VK_NULL_HANDLE, 1, &pipeInfo, nullptr, &pipe); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to create graphics pipeline\n"
				"vkCreateGraphicsPipelines() returned %s", lut::to_string(res).c_str());
		}

		return lut::Pipeline(aWindow.device, pipe);
	}

	lut::Pipeline create_pipeline(lut::VulkanWindow const& aWindow, VkRenderPass aRenderPass, VkPipelineLayout aPipelineLayout)
	{
		// Load shader modules
		lut::ShaderModule vert = lut::load_shader_module(aWindow, cfg::kVertShaderPath);
		lut::ShaderModule frag = lut::load_shader_module(aWindow, cfg::kFragShaderPath);

		// Define shader stages in the pipeline
		// Two stages, 1. Vertex shader 2. Fragment shader
		VkPipelineShaderStageCreateInfo stages[2]{};
		stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
		stages[0].module = vert.handle;
		stages[0].pName = "main";

		stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		stages[1].module = frag.handle;
		stages[1].pName = "main";

		VkVertexInputBindingDescription vertexInputs[5]{};
		// position
		vertexInputs[0].binding = 0;
		vertexInputs[0].stride = sizeof(glm::vec3);
		vertexInputs[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
		// normal
		vertexInputs[1].binding = 1;
		vertexInputs[1].stride = sizeof(glm::vec3);
		vertexInputs[1].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
		// texcoords
		vertexInputs[2].binding = 2;
		vertexInputs[2].stride = sizeof(glm::vec2);
		vertexInputs[2].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
		// colours
		vertexInputs[3].binding = 3;
		vertexInputs[3].stride = sizeof(glm::vec3);
		vertexInputs[3].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
		// surface normals
		vertexInputs[4].binding = 4;
		vertexInputs[4].stride = sizeof(glm::vec3);
		vertexInputs[4].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

		VkVertexInputAttributeDescription vertexAttributes[5]{};
		// position
		vertexAttributes[0].binding = 0;
		vertexAttributes[0].location = 0;
		vertexAttributes[0].format = VK_FORMAT_R32G32B32_SFLOAT;
		vertexAttributes[0].offset = 0;
		// normal
		vertexAttributes[1].binding = 1;
		vertexAttributes[1].location = 1;
		vertexAttributes[1].format = VK_FORMAT_R32G32B32_SFLOAT;
		vertexAttributes[1].offset = 0;
		// texcoords
		vertexAttributes[2].binding = 2;
		vertexAttributes[2].location = 2;
		vertexAttributes[2].format = VK_FORMAT_R32G32_SFLOAT;
		vertexAttributes[2].offset = 0;
		// colours
		vertexAttributes[3].binding = 3;
		vertexAttributes[3].location = 3;
		vertexAttributes[3].format = VK_FORMAT_R32G32B32_SFLOAT;
		vertexAttributes[3].offset = 0;	
		// surface normals
		vertexAttributes[4].binding = 4;
		vertexAttributes[4].location = 4;
		vertexAttributes[4].format = VK_FORMAT_R32G32B32_SFLOAT;
		vertexAttributes[4].offset = 0;

		VkPipelineVertexInputStateCreateInfo inputInfo{};
		inputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
		inputInfo.vertexBindingDescriptionCount = sizeof(vertexInputs) / sizeof(vertexInputs[0]);
		inputInfo.pVertexBindingDescriptions = vertexInputs;
		inputInfo.vertexAttributeDescriptionCount = sizeof(vertexAttributes) / sizeof(vertexAttributes[0]);
		inputInfo.pVertexAttributeDescriptions = vertexAttributes;

		// Define which primitive (point, line, triangle,...)
		VkPipelineInputAssemblyStateCreateInfo assemblyInfo{};
		assemblyInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
		assemblyInfo.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
		assemblyInfo.primitiveRestartEnable = VK_FALSE;

		// Define viewport and scissor regions
		VkViewport viewport{};
		viewport.x = 0.0f;
		viewport.y = 0.0f;
		viewport.width = float(aWindow.swapchainExtent.width);
		viewport.height = float(aWindow.swapchainExtent.height);
		viewport.minDepth = 0.0f;
		viewport.maxDepth = 1.0f;

		VkRect2D scissor{};
		scissor.offset = VkOffset2D{ 0, 0 };
		scissor.extent = VkExtent2D{ aWindow.swapchainExtent.width,
			aWindow.swapchainExtent.height };

		VkPipelineViewportStateCreateInfo viewportInfo{};
		viewportInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
		viewportInfo.viewportCount = 1;
		viewportInfo.pViewports = &viewport;
		viewportInfo.scissorCount = 1;
		viewportInfo.pScissors = &scissor;

		// Define rasterization options
		VkPipelineRasterizationStateCreateInfo rasterInfo{};
		rasterInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
		rasterInfo.depthClampEnable = VK_FALSE;
		rasterInfo.rasterizerDiscardEnable = VK_FALSE;
		rasterInfo.polygonMode = VK_POLYGON_MODE_FILL;
		rasterInfo.cullMode = VK_CULL_MODE_BACK_BIT;
		rasterInfo.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
		rasterInfo.depthBiasEnable = VK_FALSE;
		rasterInfo.lineWidth = 1.0f;

		// Define multisampling state
		VkPipelineMultisampleStateCreateInfo samplingInfo{};
		samplingInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
		samplingInfo.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

		// Define blend state
		// i.e. which color channels to write
		VkPipelineColorBlendAttachmentState blendStates[1]{};
		blendStates[0].blendEnable = VK_FALSE;
		blendStates[0].colorWriteMask = VK_COLOR_COMPONENT_R_BIT |
			VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT |
			VK_COLOR_COMPONENT_A_BIT;

		VkPipelineColorBlendStateCreateInfo blendInfo{};
		blendInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
		blendInfo.logicOpEnable = VK_FALSE;
		blendInfo.attachmentCount = 1;
		blendInfo.pAttachments = blendStates;

		// Depth Testing
		VkPipelineDepthStencilStateCreateInfo depthInfo{};
		depthInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
		depthInfo.depthTestEnable = VK_TRUE;
		depthInfo.depthWriteEnable = VK_TRUE;
		depthInfo.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
		depthInfo.minDepthBounds = 0.0f;
		depthInfo.maxDepthBounds = 1.0f;

		// Create pipeline
		VkGraphicsPipelineCreateInfo pipeInfo{};
		pipeInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;

		pipeInfo.stageCount = 2; // Vertex and fragment stages
		pipeInfo.pStages = stages;

		pipeInfo.pVertexInputState = &inputInfo;
		pipeInfo.pInputAssemblyState = &assemblyInfo;
		pipeInfo.pTessellationState = nullptr;
		pipeInfo.pViewportState = &viewportInfo;
		pipeInfo.pRasterizationState = &rasterInfo;
		pipeInfo.pMultisampleState = &samplingInfo;
		pipeInfo.pDepthStencilState = &depthInfo;
		pipeInfo.pColorBlendState = &blendInfo;
		pipeInfo.pDynamicState = nullptr;
		pipeInfo.layout = aPipelineLayout;
		pipeInfo.renderPass = aRenderPass;
		pipeInfo.subpass = 0;

		VkPipeline pipe = VK_NULL_HANDLE;
		if (auto const res = vkCreateGraphicsPipelines(aWindow.device,
			VK_NULL_HANDLE, 1, &pipeInfo, nullptr, &pipe); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to create graphics pipeline\n"
				"vkCreateGraphicsPipelines() returned %s", lut::to_string(res).c_str());
		}

		return lut::Pipeline(aWindow.device, pipe);
	}

	lut::Pipeline create_pipeline_filter_bright(lut::VulkanWindow const& aWindow, VkRenderPass aRenderPass, VkPipelineLayout aPipelineLayout)
	{
		// Load shader modules
		lut::ShaderModule vert = lut::load_shader_module(aWindow, cfg::kfilterBrightVertPath);
		lut::ShaderModule frag = lut::load_shader_module(aWindow, cfg::kfilterBrightFragPath);

		// Define shader stages in the pipeline
		// Two stages, 1. Vertex shader 2. Fragment shader
		VkPipelineShaderStageCreateInfo stages[2]{};
		stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
		stages[0].module = vert.handle;
		stages[0].pName = "main";

		stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		stages[1].module = frag.handle;
		stages[1].pName = "main";

		VkVertexInputBindingDescription vertexInputs[5]{};
		// position
		vertexInputs[0].binding = 0;
		vertexInputs[0].stride = sizeof(glm::vec3);
		vertexInputs[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
		// normal
		vertexInputs[1].binding = 1;
		vertexInputs[1].stride = sizeof(glm::vec3);
		vertexInputs[1].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
		// texcoords
		vertexInputs[2].binding = 2;
		vertexInputs[2].stride = sizeof(glm::vec2);
		vertexInputs[2].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
		// colours
		vertexInputs[3].binding = 3;
		vertexInputs[3].stride = sizeof(glm::vec3);
		vertexInputs[3].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
		// surface normals
		vertexInputs[4].binding = 4;
		vertexInputs[4].stride = sizeof(glm::vec3);
		vertexInputs[4].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

		VkVertexInputAttributeDescription vertexAttributes[5]{};
		// position
		vertexAttributes[0].binding = 0;
		vertexAttributes[0].location = 0;
		vertexAttributes[0].format = VK_FORMAT_R32G32B32_SFLOAT;
		vertexAttributes[0].offset = 0;
		// normal
		vertexAttributes[1].binding = 1;
		vertexAttributes[1].location = 1;
		vertexAttributes[1].format = VK_FORMAT_R32G32B32_SFLOAT;
		vertexAttributes[1].offset = 0;
		// texcoords
		vertexAttributes[2].binding = 2;
		vertexAttributes[2].location = 2;
		vertexAttributes[2].format = VK_FORMAT_R32G32_SFLOAT;
		vertexAttributes[2].offset = 0;
		// colours
		vertexAttributes[3].binding = 3;
		vertexAttributes[3].location = 3;
		vertexAttributes[3].format = VK_FORMAT_R32G32B32_SFLOAT;
		vertexAttributes[3].offset = 0;
		// surface normals
		vertexAttributes[4].binding = 4;
		vertexAttributes[4].location = 4;
		vertexAttributes[4].format = VK_FORMAT_R32G32B32_SFLOAT;
		vertexAttributes[4].offset = 0;

		VkPipelineVertexInputStateCreateInfo inputInfo{};
		inputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
		inputInfo.vertexBindingDescriptionCount = sizeof(vertexInputs) / sizeof(vertexInputs[0]);
		inputInfo.pVertexBindingDescriptions = vertexInputs;
		inputInfo.vertexAttributeDescriptionCount = sizeof(vertexAttributes) / sizeof(vertexAttributes[0]);
		inputInfo.pVertexAttributeDescriptions = vertexAttributes;

		// Define which primitive (point, line, triangle,...)
		VkPipelineInputAssemblyStateCreateInfo assemblyInfo{};
		assemblyInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
		assemblyInfo.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
		assemblyInfo.primitiveRestartEnable = VK_FALSE;

		// Define viewport and scissor regions
		VkViewport viewport{};
		viewport.x = 0.0f;
		viewport.y = 0.0f;
		viewport.width = float(aWindow.swapchainExtent.width);
		viewport.height = float(aWindow.swapchainExtent.height);
		viewport.minDepth = 0.0f;
		viewport.maxDepth = 1.0f;

		VkRect2D scissor{};
		scissor.offset = VkOffset2D{ 0, 0 };
		scissor.extent = VkExtent2D{ aWindow.swapchainExtent.width,
			aWindow.swapchainExtent.height };

		VkPipelineViewportStateCreateInfo viewportInfo{};
		viewportInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
		viewportInfo.viewportCount = 1;
		viewportInfo.pViewports = &viewport;
		viewportInfo.scissorCount = 1;
		viewportInfo.pScissors = &scissor;

		// Define rasterization options
		VkPipelineRasterizationStateCreateInfo rasterInfo{};
		rasterInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
		rasterInfo.depthClampEnable = VK_FALSE;
		rasterInfo.rasterizerDiscardEnable = VK_FALSE;
		rasterInfo.polygonMode = VK_POLYGON_MODE_FILL;
		rasterInfo.cullMode = VK_CULL_MODE_BACK_BIT;
		rasterInfo.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
		rasterInfo.depthBiasEnable = VK_FALSE;
		rasterInfo.lineWidth = 1.0f;

		// Define multisampling state
		VkPipelineMultisampleStateCreateInfo samplingInfo{};
		samplingInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
		samplingInfo.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

		// Define blend state
		// i.e. which color channels to write
		VkPipelineColorBlendAttachmentState blendStates[1]{};
		blendStates[0].blendEnable = VK_FALSE;
		blendStates[0].colorWriteMask = VK_COLOR_COMPONENT_R_BIT |
			VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT |
			VK_COLOR_COMPONENT_A_BIT;

		VkPipelineColorBlendStateCreateInfo blendInfo{};
		blendInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
		blendInfo.logicOpEnable = VK_FALSE;
		blendInfo.attachmentCount = 1;
		blendInfo.pAttachments = blendStates;

		// Depth Testing
		VkPipelineDepthStencilStateCreateInfo depthInfo{};
		depthInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
		depthInfo.depthTestEnable = VK_TRUE;
		depthInfo.depthWriteEnable = VK_TRUE;
		depthInfo.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
		depthInfo.minDepthBounds = 0.0f;
		depthInfo.maxDepthBounds = 1.0f;

		// Create pipeline
		VkGraphicsPipelineCreateInfo pipeInfo{};
		pipeInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;

		pipeInfo.stageCount = 2; // Vertex and fragment stages
		pipeInfo.pStages = stages;

		pipeInfo.pVertexInputState = &inputInfo;
		pipeInfo.pInputAssemblyState = &assemblyInfo;
		pipeInfo.pTessellationState = nullptr;
		pipeInfo.pViewportState = &viewportInfo;
		pipeInfo.pRasterizationState = &rasterInfo;
		pipeInfo.pMultisampleState = &samplingInfo;
		pipeInfo.pDepthStencilState = &depthInfo;
		pipeInfo.pColorBlendState = &blendInfo;
		pipeInfo.pDynamicState = nullptr;
		pipeInfo.layout = aPipelineLayout;
		pipeInfo.renderPass = aRenderPass;
		pipeInfo.subpass = 0;

		VkPipeline pipe = VK_NULL_HANDLE;
		if (auto const res = vkCreateGraphicsPipelines(aWindow.device,
			VK_NULL_HANDLE, 1, &pipeInfo, nullptr, &pipe); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to create graphics pipeline\n"
				"vkCreateGraphicsPipelines() returned %s", lut::to_string(res).c_str());
		}

		return lut::Pipeline(aWindow.device, pipe);
	}

	void create_swapchain_framebuffers(lut::VulkanWindow const& aWindow, VkRenderPass aRenderPass, 
		std::vector<lut::Framebuffer>& aFramebuffers, VkImageView aDepthView)
	{
		assert(aFramebuffers.empty());

		for (std::size_t i = 0; i < aWindow.swapViews.size(); ++i)
		{
			VkImageView attachments[2] = {
				aWindow.swapViews[i],
				aDepthView
			};

			VkFramebufferCreateInfo fbInfo{};
			fbInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
			fbInfo.flags = 0;
			fbInfo.renderPass = aRenderPass;
			fbInfo.attachmentCount = 2;
			fbInfo.pAttachments = attachments;
			fbInfo.width = aWindow.swapchainExtent.width;
			fbInfo.height = aWindow.swapchainExtent.height;
			fbInfo.layers = 1;

			VkFramebuffer fb = VK_NULL_HANDLE;
			if (auto const res = vkCreateFramebuffer(aWindow.device, &fbInfo,
				nullptr, &fb); VK_SUCCESS != res)
			{
				throw lut::Error("Unable to create framebuffer for swap chain"
					"image %zu\n"
					"vkCreateFramebuffer() returned %s", i,
					lut::to_string(res).c_str());
			}

			aFramebuffers.emplace_back(lut::Framebuffer(aWindow.device, fb));
		}

		assert(aWindow.swapViews.size() == aFramebuffers.size());
	}

	std::tuple<lut::Image, lut::ImageView> create_offine_image_view(lut::VulkanWindow const& aWindow, 
		lut::Allocator const& aAllocator)
	{
		VkImageCreateInfo imageInfo{};
		imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		imageInfo.imageType = VK_IMAGE_TYPE_2D;
		imageInfo.format = aWindow.swapchainFormat;
		imageInfo.extent.width = aWindow.swapchainExtent.width;
		imageInfo.extent.height = aWindow.swapchainExtent.height;
		imageInfo.extent.depth = 1;
		imageInfo.mipLevels = 1;
		imageInfo.arrayLayers = 1;
		imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
		imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
		imageInfo.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;

		VmaAllocationCreateInfo allocInfo{};
		allocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

		VkImage image = VK_NULL_HANDLE;
		VmaAllocation allocation = VK_NULL_HANDLE;

		if (auto const res = vmaCreateImage(aAllocator.allocator, &imageInfo, &allocInfo, &image,
			&allocation, nullptr); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to create framebuffer for offline renderering"
				"vkCreateImage reutrned %s", lut::to_string(res));
		}

		lut::Image colorImage(aAllocator.allocator, image, allocation);

		VkImageViewCreateInfo viewInfo{};
		viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		viewInfo.image = colorImage.image;
		viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
		viewInfo.format = aWindow.swapchainFormat;
		viewInfo.components = VkComponentMapping{
			VK_COMPONENT_SWIZZLE_IDENTITY,
			VK_COMPONENT_SWIZZLE_IDENTITY,
			VK_COMPONENT_SWIZZLE_IDENTITY,
			VK_COMPONENT_SWIZZLE_IDENTITY
		};
		viewInfo.subresourceRange = VkImageSubresourceRange{
			VK_IMAGE_ASPECT_COLOR_BIT,
			0, 1,
			0, 1
		};

		VkImageView imageView = VK_NULL_HANDLE;

		if (auto const res = vkCreateImageView(aWindow.device, &viewInfo, nullptr, &imageView);
			VK_SUCCESS != res)
		{
			throw lut::Error("Unable to create image view for offline framebuffer"
				"vkCreateImageView returned %s\n", lut::to_string(res));
		}

		return { std::move(colorImage), lut::ImageView(aWindow.device, imageView) };
	}

	void create_framebuffer(lut::VulkanWindow const& aWindow,
		VkRenderPass aRenderPass, lut::Framebuffer& aFramebuffers, VkImageView aDepthView, VkImageView aBackView)
	{
		VkImageView attachments[2] = {
			aBackView,
			aDepthView
		};

		VkFramebufferCreateInfo fbInfo{};
		fbInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
		fbInfo.flags = 0;
		fbInfo.renderPass = aRenderPass;
		fbInfo.attachmentCount = 2;
		fbInfo.pAttachments = attachments;
		fbInfo.width = aWindow.swapchainExtent.width;
		fbInfo.height = aWindow.swapchainExtent.height;
		fbInfo.layers = 1;

		VkFramebuffer fb = VK_NULL_HANDLE;
		if (auto const res = vkCreateFramebuffer(aWindow.device, &fbInfo,
			nullptr, &fb); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to create framebuffer for swap chain\n"
				"vkCreateFramebuffer() returned %s",
				lut::to_string(res).c_str());
		}

		aFramebuffers = lut::Framebuffer(aWindow.device, fb);
	}

	void create_frame_texture(lut::VulkanWindow const& aWindow,
		VkRenderPass aRenderPass, lut::Framebuffer& aFramebuffers, VkImageView aDepthView, VkImageView aBackView)
	{
		VkImageView attachments[1] = {
			aBackView,
		};

		VkFramebufferCreateInfo fbInfo{};
		fbInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
		fbInfo.flags = 0;
		fbInfo.renderPass = aRenderPass;
		fbInfo.attachmentCount = 1;
		fbInfo.pAttachments = attachments;
		fbInfo.width = aWindow.swapchainExtent.width;
		fbInfo.height = aWindow.swapchainExtent.height;
		fbInfo.layers = 1;

		VkFramebuffer fb = VK_NULL_HANDLE;
		if (auto const res = vkCreateFramebuffer(aWindow.device, &fbInfo,
			nullptr, &fb); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to create framebuffer for swap chain\n"
				"vkCreateFramebuffer() returned %s",
				lut::to_string(res).c_str());
		}

		aFramebuffers = lut::Framebuffer(aWindow.device, fb);
	}

	void record_commands(VkCommandBuffer aCmdBuff, VkRenderPass aBackRenderPass, VkRenderPass aRenderPass, 
		VkFramebuffer aFrameBackBuffer, VkFramebuffer aBackbuffer, VkFramebuffer aFilterHorizontalBuffer,
		VkFramebuffer aFilterVerticalBuffer, VkFramebuffer aFramebuffer, 
		VkPipeline aGraphicsPipe, VkPipeline aFilterPipe, VkPipeline aHorizontalPipe, VkPipeline aVerticalPipe,
		VkPipeline aPostPipe,
		VkExtent2D const& aImageExtent, LoadedMesh& car,
		VkBuffer aSceneUBO, glsl::SceneUniform const& aSceneUniform, VkPipelineLayout aGraphicsLayout,
		VkPipelineLayout aGraphicsLayoutTexture, VkDescriptorSet aSceneDesctipror, 
		VkDescriptorSet aBackFrameBufferDescriptor,
		VkDescriptorSet aBackBufferDescriptor,
		VkDescriptorSet aFilterHorizontalDescriptor,
		VkDescriptorSet aFilterVerticalDescriptor,
		std::vector<labutils::Buffer>const& aMaterialUBOs, 
		std::vector<glsl::MaterialUniform>const& aMaterialUniforms,
		std::vector<VkDescriptorSet>const& aMaterialDescriptor,
		std::vector<labutils::Buffer>const& aMaterialPBRUBOs,
		std::vector<glsl::MaterialPBRUniform>const& aMaterialPBRUniforms,
		std::vector<VkDescriptorSet>const& aMaterialPBRDescriptor)
	{
		// Begin recording commands
		VkCommandBufferBeginInfo beginInfo{};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
		beginInfo.pInheritanceInfo = nullptr;

		if (auto const res = vkBeginCommandBuffer(aCmdBuff, &beginInfo);
			VK_SUCCESS != res)
		{
			throw lut::Error("Unable to begin recording command buffer\n"
				"vkBeginCommandBuffer() returned %s", lut::to_string(res).c_str());
		}

		// Upload scene unifrms
		lut::buffer_barrier(aCmdBuff,
			aSceneUBO,
			VK_ACCESS_UNIFORM_READ_BIT,
			VK_ACCESS_TRANSFER_WRITE_BIT,
			VK_PIPELINE_STAGE_VERTEX_SHADER_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT);

		vkCmdUpdateBuffer(aCmdBuff, aSceneUBO, 0, sizeof(glsl::SceneUniform), &aSceneUniform);

		// Make sure everything is being uploaded
		lut::buffer_barrier(aCmdBuff,
			aSceneUBO,
			VK_ACCESS_TRANSFER_WRITE_BIT,
			VK_ACCESS_UNIFORM_READ_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_PIPELINE_STAGE_VERTEX_SHADER_BIT);

		// Upload material uniforms
		for (size_t i = 0; i < aMaterialDescriptor.size(); i++)
		{
			// Normal material
			lut::buffer_barrier(aCmdBuff,
				aMaterialUBOs[i].buffer,
				VK_ACCESS_UNIFORM_READ_BIT,
				VK_ACCESS_TRANSFER_WRITE_BIT,
				VK_PIPELINE_STAGE_VERTEX_SHADER_BIT,
				VK_PIPELINE_STAGE_TRANSFER_BIT);

			vkCmdUpdateBuffer(aCmdBuff, aMaterialUBOs[i].buffer, 0,
				sizeof(glsl::MaterialUniform), &aMaterialUniforms[i]);

			// Make sure everything is being uploaded
			lut::buffer_barrier(aCmdBuff,
				aMaterialUBOs[i].buffer,
				VK_ACCESS_TRANSFER_WRITE_BIT,
				VK_ACCESS_UNIFORM_READ_BIT,
				VK_PIPELINE_STAGE_TRANSFER_BIT,
				VK_PIPELINE_STAGE_VERTEX_SHADER_BIT);

			// PBR material
			lut::buffer_barrier(aCmdBuff,
				aMaterialPBRUBOs[i].buffer,
				VK_ACCESS_UNIFORM_READ_BIT,
				VK_ACCESS_TRANSFER_WRITE_BIT,
				VK_PIPELINE_STAGE_VERTEX_SHADER_BIT,
				VK_PIPELINE_STAGE_TRANSFER_BIT);

			vkCmdUpdateBuffer(aCmdBuff, aMaterialPBRUBOs[i].buffer, 0,
				sizeof(glsl::MaterialPBRUniform), &aMaterialPBRUniforms[i]);

			// Make sure everything is being uploaded
			lut::buffer_barrier(aCmdBuff,
				aMaterialPBRUBOs[i].buffer,
				VK_ACCESS_TRANSFER_WRITE_BIT,
				VK_ACCESS_UNIFORM_READ_BIT,
				VK_PIPELINE_STAGE_TRANSFER_BIT,
				VK_PIPELINE_STAGE_VERTEX_SHADER_BIT);
		}

		// Begin render pass
		VkClearValue clearValues[2]{};
		clearValues[0].color.float32[0] = 0.0f; // Clear to a dark gray background
		clearValues[0].color.float32[1] = 0.0f;
		clearValues[0].color.float32[2] = 0.0f;
		clearValues[0].color.float32[3] = 1.0f;
		// Depth
		clearValues[1].depthStencil.depth = 1.0f;

		VkRenderPassBeginInfo backPassInfo{};
		backPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
		backPassInfo.renderPass = aBackRenderPass;
		backPassInfo.framebuffer = aBackbuffer;
		backPassInfo.renderArea.offset = VkOffset2D{ 0, 0 };
		backPassInfo.renderArea.extent = aImageExtent;
		backPassInfo.clearValueCount = 2;
		backPassInfo.pClearValues = clearValues;

		vkCmdBeginRenderPass(aCmdBuff, &backPassInfo, VK_SUBPASS_CONTENTS_INLINE);

		// Bind descriptor set
		vkCmdBindDescriptorSets(aCmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, aGraphicsLayout,
			0, 1, &aSceneDesctipror, 0, nullptr);

		// Render the brightest part first

		// Bind every vertex buff and draw it
		for (size_t i = 0; i < car.positions.size(); i++)
		{
			vkCmdBindDescriptorSets(aCmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, aGraphicsLayout,
				1, 1, &aMaterialDescriptor[car.materialIndex[i]], 0, nullptr);

			vkCmdBindDescriptorSets(aCmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, aGraphicsLayout,
				2, 1, &aMaterialPBRDescriptor[car.materialIndex[i]], 0, nullptr);

			// Begin drawing with our graphics pipeline
			// If there is no texture use the normal graphics pipeline
			vkCmdBindPipeline(aCmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, aFilterPipe);

			VkBuffer buffers[5] = { car.positions[i].buffer, car.normals[i].buffer, car.texCorods[i].buffer,
				car.colors[i].buffer, car.surfaceNormals[i].buffer};
			VkDeviceSize offsets[5]{};

			vkCmdBindVertexBuffers(aCmdBuff, 0, sizeof(buffers) / sizeof(buffers[0]), buffers, offsets);

			vkCmdDraw(aCmdBuff, car.vertexCount[i], 1, 0, 0);
		}

		// End the render pass
		vkCmdEndRenderPass(aCmdBuff);


		// Gaussian Blur
		// Horizontal first
		backPassInfo.framebuffer = aFilterHorizontalBuffer;
		//backPassInfo.renderPass = aRenderPass;

		vkCmdBeginRenderPass(aCmdBuff, &backPassInfo, VK_SUBPASS_CONTENTS_INLINE);

		// Bind pipeline
		vkCmdBindPipeline(aCmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, aHorizontalPipe);
		// Bind descriptor set
		vkCmdBindDescriptorSets(aCmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, aGraphicsLayoutTexture,
			0, 1, &aBackBufferDescriptor, 0, nullptr);

		// Bind texture
		vkCmdDraw(aCmdBuff, 3, 1, 0, 0);

		// End the render pass
		vkCmdEndRenderPass(aCmdBuff);


		clearValues[0].color.float32[0] = 0.1f; // Clear to a dark gray background
		clearValues[0].color.float32[1] = 0.1f;
		clearValues[0].color.float32[2] = 0.1f;
		clearValues[0].color.float32[3] = 1.0f;
		backPassInfo.pClearValues = clearValues;

		// Now Vertical
		backPassInfo.framebuffer = aFilterVerticalBuffer;

		vkCmdBeginRenderPass(aCmdBuff, &backPassInfo, VK_SUBPASS_CONTENTS_INLINE);

		// Bind pipeline
		vkCmdBindPipeline(aCmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, aVerticalPipe);
		// Bind descriptor set
		vkCmdBindDescriptorSets(aCmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, aGraphicsLayoutTexture,
			0, 1, &aFilterHorizontalDescriptor, 0, nullptr);

		// Bind texture
		vkCmdDraw(aCmdBuff, 3, 1, 0, 0);

		// End the render pass
		vkCmdEndRenderPass(aCmdBuff);



		// Render the actual Scene
		backPassInfo.framebuffer = aFrameBackBuffer;

		vkCmdBeginRenderPass(aCmdBuff, &backPassInfo, VK_SUBPASS_CONTENTS_INLINE);

		// Bind descriptor set
		vkCmdBindDescriptorSets(aCmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, aGraphicsLayout,
			0, 1, &aSceneDesctipror, 0, nullptr);

		// Bind every vertex buff and draw it
		for (size_t i = 0; i < car.positions.size(); i++)
		{
			vkCmdBindDescriptorSets(aCmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, aGraphicsLayout,
				1, 1, &aMaterialDescriptor[car.materialIndex[i]], 0, nullptr);

			vkCmdBindDescriptorSets(aCmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, aGraphicsLayout,
				2, 1, &aMaterialPBRDescriptor[car.materialIndex[i]], 0, nullptr);

			// Begin drawing with our graphics pipeline
			// If there is no texture use the normal graphics pipeline
			vkCmdBindPipeline(aCmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, aGraphicsPipe);

			VkBuffer buffers[5] = { car.positions[i].buffer, car.normals[i].buffer, car.texCorods[i].buffer,
				car.colors[i].buffer, car.surfaceNormals[i].buffer };
			VkDeviceSize offsets[5]{};

			vkCmdBindVertexBuffers(aCmdBuff, 0, sizeof(buffers) / sizeof(buffers[0]), buffers, offsets);

			vkCmdDraw(aCmdBuff, car.vertexCount[i], 1, 0, 0);
		}

		// End the render pass
		vkCmdEndRenderPass(aCmdBuff);

		VkRenderPassBeginInfo passInfo{};
		passInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
		passInfo.renderPass = aRenderPass;
		passInfo.framebuffer = aFramebuffer;
		passInfo.renderArea.offset = VkOffset2D{ 0, 0 };
		passInfo.renderArea.extent = aImageExtent;
		passInfo.clearValueCount = 2;
		passInfo.pClearValues = clearValues;

		vkCmdBeginRenderPass(aCmdBuff, &passInfo, VK_SUBPASS_CONTENTS_INLINE);

		// Bind pipeline
		vkCmdBindPipeline(aCmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, aPostPipe);
		// Bind descriptor set
		vkCmdBindDescriptorSets(aCmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, aGraphicsLayoutTexture,
			0, 1, &aFilterVerticalDescriptor, 0, nullptr);
		vkCmdBindDescriptorSets(aCmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, aGraphicsLayoutTexture,
			1, 1, &aBackFrameBufferDescriptor, 0, nullptr);

		// Bind texture
		vkCmdDraw(aCmdBuff, 3, 1, 0, 0);

		// End the render pass
		vkCmdEndRenderPass(aCmdBuff);


		// End command recording
		if (auto const res = vkEndCommandBuffer(aCmdBuff); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to end recording command buffer\n"
				"vkEndCommandBuffer() returned %s", lut::to_string(res).c_str());
		}
	}

	void post_processing(VkCommandBuffer aCmdBuff, VkRenderPass aRenderPass, VkFramebuffer aFramebuffer,
		VkPipeline aGraphicsPipe, VkExtent2D const& aImageExtent)
	{
		// Begin recording commands
		VkCommandBufferBeginInfo beginInfo{};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
		beginInfo.pInheritanceInfo = nullptr;

		if (auto const res = vkBeginCommandBuffer(aCmdBuff, &beginInfo);
			VK_SUCCESS != res)
		{
			throw lut::Error("Unable to begin recording command buffer\n"
				"vkBeginCommandBuffer() returned %s", lut::to_string(res).c_str());
		}

		// Begin render pass
		VkClearValue clearValues[2]{};
		clearValues[0].color.float32[0] = 0.1f; // Clear to a dark gray background
		clearValues[0].color.float32[1] = 0.1f;
		clearValues[0].color.float32[2] = 0.1f;
		clearValues[0].color.float32[3] = 1.0f;
		// Depth
		clearValues[1].depthStencil.depth = 1.0f;

		VkRenderPassBeginInfo passInfo{};
		passInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
		passInfo.renderPass = aRenderPass;
		passInfo.framebuffer = aFramebuffer;
		passInfo.renderArea.offset = VkOffset2D{ 0, 0 };
		passInfo.renderArea.extent = aImageExtent;
		passInfo.clearValueCount = 2;
		passInfo.pClearValues = clearValues;

		vkCmdBeginRenderPass(aCmdBuff, &passInfo, VK_SUBPASS_CONTENTS_INLINE);

		// End the render pass
		vkCmdEndRenderPass(aCmdBuff);

		// End command recording
		if (auto const res = vkEndCommandBuffer(aCmdBuff); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to end recording command buffer\n"
				"vkEndCommandBuffer() returned %s", lut::to_string(res).c_str());
		}
	}

	void submit_commands(lut::VulkanContext const& aContext, VkCommandBuffer aCmdBuff, VkFence aFence, VkSemaphore aWaitSemaphore, VkSemaphore aSignalSemaphore)
	{
		VkPipelineStageFlags waitPipelineStages =
			VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;

		VkSubmitInfo submitInfo{};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &aCmdBuff;

		submitInfo.waitSemaphoreCount = 1;
		submitInfo.pWaitSemaphores = &aWaitSemaphore;
		submitInfo.pWaitDstStageMask = &waitPipelineStages;

		submitInfo.signalSemaphoreCount = 1;
		submitInfo.pSignalSemaphores = &aSignalSemaphore;

		if (auto const res = vkQueueSubmit(aContext.graphicsQueue, 1, &submitInfo,
			aFence); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to submit command buffer to queue\n"
				"vkQueueSubmit() returned %s", lut::to_string(res).c_str());
		}
	}

	std::tuple<lut::Image, lut::ImageView> create_depth_buffer(lut::VulkanWindow const& aWindow, lut::Allocator const& aAllocator)
	{
		VkImageCreateInfo imageInfo{};
		imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		imageInfo.imageType = VK_IMAGE_TYPE_2D;
		imageInfo.format = cfg::kDepthFormat;
		imageInfo.extent.width = aWindow.swapchainExtent.width;
		imageInfo.extent.height = aWindow.swapchainExtent.height;
		imageInfo.extent.depth = 1;
		imageInfo.mipLevels = 1;
		imageInfo.arrayLayers = 1;
		imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
		imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
		imageInfo.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
		imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
		imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

		VmaAllocationCreateInfo allocInfo{};
		allocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

		VkImage image = VK_NULL_HANDLE;
		VmaAllocation allocation = VK_NULL_HANDLE;

		if (auto const res = vmaCreateImage(aAllocator.allocator, &imageInfo, &allocInfo, &image,
			&allocation, nullptr); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to allocate depth buffer image.\n"
				"vmaCreateImage() returned %s", lut::to_string(res));
		}

		lut::Image depthImage(aAllocator.allocator, image, allocation);

		// Create the image view
		VkImageViewCreateInfo viewInfo{};
		viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		viewInfo.image = depthImage.image;
		viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
		viewInfo.format = cfg::kDepthFormat;
		viewInfo.components = VkComponentMapping{};
		viewInfo.subresourceRange = VkImageSubresourceRange{
			VK_IMAGE_ASPECT_DEPTH_BIT,
			0, 1,
			0, 1
		};
		VkImageView view = VK_NULL_HANDLE;
		if (auto const res = vkCreateImageView(aWindow.device, &viewInfo, nullptr, &view);
			VK_SUCCESS != res)
		{
			throw lut::Error("Unable to create image view\n"
				"vkCreateImageView() returned %s", lut::to_string(res).c_str());
		}

		return { std::move(depthImage), lut::ImageView(aWindow.device, view) };
	}

	void updateBackBufferDescriptorSet(lut::VulkanWindow const& aWindow, VkDescriptorSet const& backBufferDescriptor,
		VkImageView const& backBufferView, VkSampler const& filterSampler)
	{
		{
			VkWriteDescriptorSet desc[1]{};

			VkDescriptorImageInfo textureInfo{};
			textureInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			textureInfo.imageView = backBufferView;
			textureInfo.sampler = filterSampler;

			desc[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			desc[0].dstSet = backBufferDescriptor;
			desc[0].dstBinding = 0;
			desc[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
			desc[0].descriptorCount = 1;
			desc[0].pImageInfo = &textureInfo;

			constexpr auto numSets = sizeof(desc) / sizeof(desc[0]);
			vkUpdateDescriptorSets(aWindow.device, numSets, desc, 0, nullptr);
		}
	}
}

//EOF vim:syntax=cpp:foldmethod=marker:ts=4:noexpandtab: 