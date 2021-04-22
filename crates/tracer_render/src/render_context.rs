use std::ffi::CString;
use std::sync::Arc;

use ash::{Device, Instance, vk};
use ash::extensions::khr::{AccelerationStructure, Swapchain};
use ash::version::{DeviceV1_0, DeviceV1_2, InstanceV1_1};
use bevy::utils::tracing::*;
use crevice::std430::{AsStd430, Std430};

use crate::mesh::{Mesh, Vertex};
use crate::renderer::{Renderer, VulkanContext};
use crate::swapchain::SwapchainDescriptor;
use crate::vk_types::AllocatedImage;

#[derive(Copy, Clone, AsStd430)]
pub struct MeshPushConstants {
    pub data: mint::Vector3<f32>,
    pub matrix: mint::ColumnMatrix4<f32>,
}

struct FrameSync {
    image_available_semaphore: vk::Semaphore,
    render_finished_semaphore: vk::Semaphore,
    fence: vk::Fence,
}

impl FrameSync {
    fn new(device: &Device) -> Self {
        let image_available_semaphore = unsafe {
            device
                .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
                .unwrap()
        };
        let render_finished_semaphore = unsafe {
            device
                .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
                .unwrap()
        };
        let fence = unsafe {
            let create_info = vk::FenceCreateInfo::builder().flags(vk::FenceCreateFlags::SIGNALED);
            device.create_fence(&create_info, None).unwrap()
        };
        FrameSync {
            image_available_semaphore,
            render_finished_semaphore,
            fence,
        }
    }
}

pub struct RenderContext {
    vk_context: Arc<VulkanContext>,
    allocator: Arc<vk_mem::Allocator>,
    surface: vk::SurfaceKHR,
    swapchain_loader: Swapchain,
    swapchain: vk::SwapchainKHR,

    swapchain_config: SwapchainDescriptor,
    swapchain_image_views: Vec<vk::ImageView>,
    depth_image: AllocatedImage,
    depth_image_view: vk::ImageView,

    frame_sync: Vec<FrameSync>,

    graphics_queue: vk::Queue,
    command_pool: vk::CommandPool,
    render_pass: vk::RenderPass,
    framebuffers: Vec<vk::Framebuffer>,
    command_buffers: Vec<vk::CommandBuffer>,

    total_frame_count: usize,

    graphics_pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,

    acceleration_structure_loader: AccelerationStructure,
    meshes: Vec<Mesh>,
}

impl RenderContext {
    pub fn new(renderer: &Renderer) -> Self {
        let device = &renderer.vk_context.device;
        let surface = renderer.surface;
        let graphics_queue = renderer.graphics_queue;
        let vk_context = renderer.vk_context.clone();

        let allocator = create_vulkan_allocator(device, &renderer.vk_context.instance, renderer.vk_context.physical_device);
        let allocator = Arc::new(allocator);

        let swapchain_config = SwapchainDescriptor::new(renderer.vk_context.physical_device, &renderer.surface_loader, surface);
        let swapchain_loader = Swapchain::new(&renderer.vk_context.instance, device);

        let swapchain = create_swapchain(&swapchain_loader, surface, &swapchain_config, None);
        let swapchain_images = unsafe { swapchain_loader.get_swapchain_images(swapchain).unwrap() };

        let command_pool = create_command_pool(device, renderer.graphics_queue_family);
        let command_buffers =
            create_command_buffers(device, &command_pool, swapchain_images.len() as u32);
        let render_pass = create_default_render_pass(&device, swapchain_config.format);

        let swapchain_image_views =
            create_swapchain_image_views(device, &swapchain_images, &swapchain_config);
        let (depth_image, depth_image_view) = create_depth_image(&device, &allocator, swapchain_config.extent);
        let framebuffers = create_framebuffers(
            &device,
            render_pass,
            &swapchain_image_views,
            depth_image_view,
            swapchain_config.extent,
            swapchain_images.len(),
        );

        let frame_sync = {
            (0..swapchain_config.frames_in_flight).map(|_| { FrameSync::new(&device) }).collect()
        };

        let (graphics_pipeline, pipeline_layout) = create_graphics_pipeline(device, render_pass);

        let mesh = Mesh::load_from_obj("assets/models/monkey.obj", &allocator);
        let meshes = vec![mesh];

        let _raytracing_properties = get_physical_device_properties(&vk_context.instance, vk_context.physical_device);
        let acceleration_structure_loader = AccelerationStructure::new(&vk_context.instance, &vk_context.device);

        let blas = create_bottom_level_acceleration_structures(device, &acceleration_structure_loader, &meshes);

        RenderContext {
            vk_context,
            allocator,
            surface,
            swapchain_loader,
            swapchain,
            swapchain_config,
            swapchain_image_views,
            depth_image,
            depth_image_view,
            frame_sync,
            graphics_queue,
            command_pool,
            render_pass,
            framebuffers,
            command_buffers,
            total_frame_count: 0,
            graphics_pipeline,
            pipeline_layout,
            acceleration_structure_loader,
            meshes,
        }
    }

    pub fn recreate_swapchain(&mut self, width: f32, height: f32) {
        let device = &self.vk_context.device;
        unsafe {
            // delete old resources
            device.device_wait_idle().unwrap();
            for &framebuffer in self.framebuffers.iter() {
                device.destroy_framebuffer(framebuffer, None);
            }
            for &swapchain_image_view in self.swapchain_image_views.iter() {
                device.destroy_image_view(swapchain_image_view, None);
            }
            self.depth_image.destroy(&self.allocator);
            device.destroy_image_view(self.depth_image_view, None);
            device.reset_command_pool(self.command_pool, vk::CommandPoolResetFlags::empty()).unwrap()
        }

        self.swapchain_config.extent.width = width as u32;
        self.swapchain_config.extent.height = height as u32;
        let old_swapchain = self.swapchain;
        self.swapchain = create_swapchain(&self.swapchain_loader, self.surface, &self.swapchain_config, Some(old_swapchain));
        unsafe {
            self.swapchain_loader.destroy_swapchain(old_swapchain, None);
        }

        let swapchain_images = unsafe { self.swapchain_loader.get_swapchain_images(self.swapchain).unwrap() };
        self.swapchain_image_views =
            create_swapchain_image_views(device, &swapchain_images, &self.swapchain_config);
        let (depth_image, depth_image_view) = create_depth_image(&device, &self.allocator, self.swapchain_config.extent);
        self.depth_image = depth_image;
        self.depth_image_view = depth_image_view;
        self.framebuffers = create_framebuffers(
            &device,
            self.render_pass,
            &self.swapchain_image_views,
            depth_image_view,
            self.swapchain_config.extent,
            swapchain_images.len(),
        );
    }

    pub unsafe fn draw(&mut self) {
        let device = &self.vk_context.device;
        let current_frame = self.total_frame_count % self.swapchain_config.frames_in_flight as usize;
        let frame_sync = &self.frame_sync[current_frame];

        device.wait_for_fences(&[frame_sync.fence], true, u64::MAX).unwrap();

        let image_index;
        let next_swapchain_image = {
            self.swapchain_loader.acquire_next_image(self.swapchain, u64::MAX, frame_sync.image_available_semaphore, vk::Fence::null())
        };
        match next_swapchain_image {
            Ok((index, _is_suboptimal)) => image_index = index as usize,
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => return,
            Err(error) => panic!("Failed to acquire swapchain image: {}", error),
        }

        device.reset_fences(&[frame_sync.fence]).unwrap();

        let clear_values = [
            vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0.5, 0.25, 0.25, 0.0],
                },
            },
            vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue {
                    depth: 1.0,
                    stencil: 0,
                },
            },
        ];

        let command_buffer = self.command_buffers[image_index];

        device
            .begin_command_buffer(command_buffer, &vk::CommandBufferBeginInfo::builder())
            .unwrap();

        let render_pass_begin_info = vk::RenderPassBeginInfo::builder()
            .render_pass(self.render_pass)
            .framebuffer(self.framebuffers[image_index])
            .render_area(
                vk::Rect2D::builder()
                    .offset(vk::Offset2D::builder().x(0).y(0).build())
                    .extent(self.swapchain_config.extent)
                    .build(),
            )
            .clear_values(&clear_values);

        device.cmd_begin_render_pass(
            command_buffer,
            &render_pass_begin_info,
            vk::SubpassContents::INLINE,
        );

        device.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::GRAPHICS, self.graphics_pipeline);

        // We flip the viewport y axis so that it points up
        // https://www.saschawillems.de/blog/2019/03/29/flipping-the-vulkan-viewport/
        let extent = self.swapchain_config.extent;
        device.cmd_set_viewport(
            command_buffer,
            0,
            &[vk::Viewport::builder()
                .x(0.0)
                .y(extent.height as f32)
                .width(extent.width as f32)
                .height(-(extent.height as f32))
                .min_depth(0.0)
                .max_depth(1.0)
                .build()],
        );
        device.cmd_set_scissor(
            command_buffer,
            0,
            &[vk::Rect2D::builder()
                .offset(vk::Offset2D::default())
                .extent(extent)
                .build()],
        );

        // device.cmd_push_constants(
        //     command_buffer,
        //     mat.pipeline_layout,
        //     vk::ShaderStageFlags::VERTEX,
        //     0,
        //     constants.as_std430().as_bytes(),
        // );
        //
        // self.device
        //     .cmd_draw(command_buffer, mesh.vertices.len() as u32, 1, 0, 0);

        device.cmd_end_render_pass(command_buffer);

        device.end_command_buffer(command_buffer).unwrap();

        let submit_info = vk::SubmitInfo::builder()
            .wait_dst_stage_mask(&[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT])
            .command_buffers(&[command_buffer])
            .wait_semaphores(&[frame_sync.image_available_semaphore])
            .signal_semaphores(&[frame_sync.render_finished_semaphore])
            .build();

        device
            .queue_submit(self.graphics_queue, &[submit_info], frame_sync.fence)
            .unwrap();

        let present_info = vk::PresentInfoKHR::builder()
            .swapchains(&[self.swapchain])
            .image_indices(&[image_index as u32])
            .wait_semaphores(&[frame_sync.render_finished_semaphore])
            .build();
        let present_result = self
            .swapchain_loader
            .queue_present(self.graphics_queue, &present_info);
        match present_result {
            Ok(_) => (),
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => (),
            Err(error) => panic!("Failed to present queue: {}", error),
        }

        self.total_frame_count += 1;
    }
}

impl Drop for RenderContext {
    fn drop(&mut self) {
        unsafe {
            let device = &self.vk_context.device;

            for &swapchain_image_view in self.swapchain_image_views.iter() {
                device.destroy_image_view(swapchain_image_view, None);
            }
            self.depth_image.destroy(&self.allocator);
            device.destroy_image_view(self.depth_image_view, None);
            for &framebuffer in self.framebuffers.iter() {
                device.destroy_framebuffer(framebuffer, None);
            }

            device.destroy_render_pass(self.render_pass, None);
            for frame_sync in self.frame_sync.iter() {
                device.destroy_fence(frame_sync.fence, None);
                device.destroy_semaphore(frame_sync.image_available_semaphore, None);
                device.destroy_semaphore(frame_sync.render_finished_semaphore, None);
            }

            device.destroy_pipeline(self.graphics_pipeline, None);
            device.destroy_pipeline_layout(self.pipeline_layout, None);

            self.swapchain_loader.destroy_swapchain(self.swapchain, None);
            device.destroy_command_pool(self.command_pool, None);
        }
    }
}

fn create_shader_module(device: &Device, file: &str) -> vk::ShaderModule {
    use std::fs::File;
    use std::path::Path;
    let mut spv_file = File::open(&Path::new(file)).expect("Shader file not found");
    let spirv = ash::util::read_spv(&mut spv_file).unwrap();
    let create_info = vk::ShaderModuleCreateInfo::builder().code(&spirv);
    unsafe { device.create_shader_module(&create_info, None).unwrap() }
}

fn create_graphics_pipeline(
    device: &Device,
    render_pass: vk::RenderPass,
) -> (vk::Pipeline, vk::PipelineLayout) {
    let vertex_shader_module = create_shader_module(device, "assets/shaders/triangle.vert.spv");
    let fragment_shader_module = create_shader_module(device, "assets/shaders/triangle.frag.spv");
    let shader_entry_point = CString::new("main").unwrap();

    let pipeline_shader_stages = [
        vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vertex_shader_module)
            .name(&shader_entry_point)
            .build(),
        vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(fragment_shader_module)
            .name(&shader_entry_point)
            .build(),
    ];

    let push_constant = vk::PushConstantRange::builder()
        .offset(0)
        .size(std::mem::size_of::<Std430MeshPushConstants>() as u32)
        .stage_flags(vk::ShaderStageFlags::VERTEX)
        .build();

    let pipeline_layout = {
        let create_info = vk::PipelineLayoutCreateInfo::builder()
            .push_constant_ranges(&[push_constant])
            .build();
        unsafe { device.create_pipeline_layout(&create_info, None) }.unwrap()
    };

    let input_assembly_info = vk::PipelineInputAssemblyStateCreateInfo::builder()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
    let viewport_info = vk::PipelineViewportStateCreateInfo::builder()
        .viewport_count(1)
        .scissor_count(1);
    let rasterization_info = vk::PipelineRasterizationStateCreateInfo::builder()
        .depth_clamp_enable(false)
        .rasterizer_discard_enable(false)
        .polygon_mode(vk::PolygonMode::FILL)
        .cull_mode(vk::CullModeFlags::BACK)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .depth_bias_enable(false)
        .line_width(1.0);

    let stencil_op = vk::StencilOpState::builder()
        .fail_op(vk::StencilOp::KEEP)
        .pass_op(vk::StencilOp::KEEP)
        .compare_op(vk::CompareOp::ALWAYS)
        .build();
    let depth_stencil_info = vk::PipelineDepthStencilStateCreateInfo::builder()
        .depth_test_enable(true)
        .depth_write_enable(true)
        .depth_compare_op(vk::CompareOp::LESS_OR_EQUAL)
        .depth_bounds_test_enable(false)
        .stencil_test_enable(false)
        .front(stencil_op)
        .back(stencil_op);
    let color_blend_attachments = [vk::PipelineColorBlendAttachmentState::builder()
        .color_write_mask(
            vk::ColorComponentFlags::R
                | vk::ColorComponentFlags::G
                | vk::ColorComponentFlags::B
                | vk::ColorComponentFlags::A,
        )
        .build()];
    let color_blend_info =
        vk::PipelineColorBlendStateCreateInfo::builder().attachments(&color_blend_attachments);

    let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic_state_info =
        vk::PipelineDynamicStateCreateInfo::builder().dynamic_states(&dynamic_states);

    let vertex_input_attributes = [];//Vertex::get_attribute_descriptions();
    let vertex_input_bindings = [];//Vertex::get_binding_descriptions();
    let vertex_input_state = vk::PipelineVertexInputStateCreateInfo::builder()
        .vertex_attribute_descriptions(&vertex_input_attributes)
        .vertex_binding_descriptions(&vertex_input_bindings);

    let multisample_info = vk::PipelineMultisampleStateCreateInfo::builder()
        .rasterization_samples(vk::SampleCountFlags::TYPE_1);

    let pipeline_create_info = [vk::GraphicsPipelineCreateInfo::builder()
        .stages(&pipeline_shader_stages)
        .vertex_input_state(&vertex_input_state)
        .input_assembly_state(&input_assembly_info)
        .viewport_state(&viewport_info)
        .rasterization_state(&rasterization_info)
        .multisample_state(&multisample_info)
        .depth_stencil_state(&depth_stencil_info)
        .color_blend_state(&color_blend_info)
        .dynamic_state(&dynamic_state_info)
        .layout(pipeline_layout)
        .render_pass(render_pass)
        .subpass(0)
        .build()];

    let graphics_pipeline = unsafe {
        device
            .create_graphics_pipelines(vk::PipelineCache::null(), &pipeline_create_info, None)
            .unwrap()[0]
    };

    unsafe {
        device.destroy_shader_module(vertex_shader_module, None);
        device.destroy_shader_module(fragment_shader_module, None);
    }

    (graphics_pipeline, pipeline_layout)
}

fn create_vulkan_allocator(
    device: &Device,
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
) -> vk_mem::Allocator {
    let create_info = vk_mem::AllocatorCreateInfo {
        physical_device,
        device: device.clone(),
        instance: instance.clone(),
        flags: 0x00000020.fr,
        preferred_large_heap_block_size: 0,
        frame_in_use_count: 0,
        heap_size_limits: None,
    };
    vk_mem::Allocator::new(&create_info).unwrap()
}

fn create_swapchain(
    swapchain_loader: &Swapchain,
    surface: vk::SurfaceKHR,
    swapchain_config: &SwapchainDescriptor,
    old_swapchain: Option<vk::SwapchainKHR>,
) -> vk::SwapchainKHR {
    let swapchain_create_info = vk::SwapchainCreateInfoKHR::builder()
        .surface(surface)
        .min_image_count(swapchain_config.frames_in_flight)
        .image_format(swapchain_config.format)
        .image_color_space(vk::ColorSpaceKHR::SRGB_NONLINEAR)
        .image_extent(swapchain_config.extent)
        .image_array_layers(1)
        .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
        .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
        .pre_transform(vk::SurfaceTransformFlagsKHR::IDENTITY)
        .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
        .present_mode(vk::PresentModeKHR::MAILBOX)
        .clipped(true);

    let swapchain_create_info = match old_swapchain {
        None => swapchain_create_info,
        Some(old_swapchain) => swapchain_create_info.old_swapchain(old_swapchain),
    };

    unsafe {
        swapchain_loader
            .create_swapchain(&swapchain_create_info, None)
            .unwrap()
    }
}

fn create_command_pool(device: &Device, queue_index: u32) -> vk::CommandPool {
    let command_pool_create_info = vk::CommandPoolCreateInfo::builder()
        .queue_family_index(queue_index)
        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
        .build();
    unsafe {
        device
            .create_command_pool(&command_pool_create_info, None)
            .unwrap()
    }
}

fn create_command_buffers(
    device: &Device,
    command_pool: &vk::CommandPool,
    count: u32,
) -> Vec<vk::CommandBuffer> {
    let create_info = vk::CommandBufferAllocateInfo::builder()
        .command_pool(*command_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(count);
    unsafe { device.allocate_command_buffers(&create_info).unwrap() }
}

fn create_default_render_pass(device: &Device, format: vk::Format) -> vk::RenderPass {
    let attachments = [
        // Color
        vk::AttachmentDescription::builder()
            .format(format)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::PRESENT_SRC_KHR)
            .build(),
        // Depth
        vk::AttachmentDescription::builder()
            .format(vk::Format::D32_SFLOAT)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::DONT_CARE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
            .build(),
    ];

    let color_reference = [vk::AttachmentReference::builder()
        .attachment(0)
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
        .build()];
    let depth_reference = vk::AttachmentReference::builder()
        .attachment(1)
        .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
        .build();
    let subpasses = [vk::SubpassDescription::builder()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(&color_reference)
        .depth_stencil_attachment(&depth_reference)
        .build()];
    let render_pass_create_info = vk::RenderPassCreateInfo::builder()
        .attachments(&attachments)
        .subpasses(&subpasses)
        .build();
    unsafe {
        device
            .create_render_pass(&render_pass_create_info, None)
            .unwrap()
    }
}

fn create_swapchain_image_views(
    device: &Device,
    swapchain_images: &[vk::Image],
    swapchain_config: &SwapchainDescriptor,
) -> Vec<vk::ImageView> {
    swapchain_images
        .iter()
        .map(|&image| {
            let image_view_create_info = vk::ImageViewCreateInfo::builder()
                .image(image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(swapchain_config.format)
                .subresource_range(
                    vk::ImageSubresourceRange::builder()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .base_mip_level(0)
                        .level_count(1)
                        .base_array_layer(0)
                        .layer_count(1)
                        .build(),
                );
            unsafe {
                device
                    .create_image_view(&image_view_create_info, None)
                    .unwrap()
            }
        })
        .collect()
}

fn create_depth_image(
    device: &Device,
    allocator: &vk_mem::Allocator,
    extent: vk::Extent2D,
) -> (AllocatedImage, vk::ImageView) {
    let depth_image_create_info = vk::ImageCreateInfo::builder()
        .format(vk::Format::D32_SFLOAT)
        .samples(vk::SampleCountFlags::TYPE_1)
        .mip_levels(1)
        .array_layers(1)
        .usage(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .image_type(vk::ImageType::TYPE_2D)
        .extent(vk::Extent3D {
            width: extent.width,
            height: extent.height,
            depth: 1,
        })
        .build();

    let depth_image_allocation_info = vk_mem::AllocationCreateInfo {
        usage: vk_mem::MemoryUsage::GpuOnly,
        flags: vk_mem::AllocationCreateFlags::empty(),
        required_flags: vk::MemoryPropertyFlags::DEVICE_LOCAL,
        preferred_flags: vk::MemoryPropertyFlags::empty(),
        memory_type_bits: 0,
        pool: None,
        user_data: None,
    };

    let depth_allocated_image = AllocatedImage::new(
        allocator,
        depth_image_create_info,
        depth_image_allocation_info,
    );

    let depth_image_view = {
        let depth_image_view_create_info = vk::ImageViewCreateInfo::builder()
            .image(depth_allocated_image.image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(vk::Format::D32_SFLOAT)
            .subresource_range(
                vk::ImageSubresourceRange::builder()
                    .aspect_mask(vk::ImageAspectFlags::DEPTH)
                    .base_mip_level(0)
                    .level_count(1)
                    .base_array_layer(0)
                    .layer_count(1)
                    .build(),
            );
        unsafe {
            device
                .create_image_view(&depth_image_view_create_info, None)
                .unwrap()
        }
    };

    (depth_allocated_image, depth_image_view)
}

fn create_framebuffers(
    device: &Device,
    render_pass: vk::RenderPass,
    swapchain_image_views: &[vk::ImageView],
    depth_image_view: vk::ImageView,
    extent: vk::Extent2D,
    count: usize,
) -> Vec<vk::Framebuffer> {
    (0..count)
        .map(|i| {
            let attachments = [swapchain_image_views[i], depth_image_view];
            let framebuffer_create_info = vk::FramebufferCreateInfo::builder()
                .render_pass(render_pass)
                .attachments(&attachments)
                .width(extent.width)
                .height(extent.height)
                .layers(1);
            unsafe {
                device
                    .create_framebuffer(&framebuffer_create_info, None)
                    .unwrap()
            }
        }
        ).collect()
}

fn get_physical_device_properties(instance: &ash::Instance, physical_device: vk::PhysicalDevice) -> vk::PhysicalDeviceProperties2 {
    let mut physical_device_properties = vk::PhysicalDeviceProperties2::default();
    unsafe { instance.get_physical_device_properties2(physical_device, &mut physical_device_properties) }
    physical_device_properties
}

struct BlasInput {
    pub as_geometry: Vec<vk::AccelerationStructureGeometryKHR>,
    pub as_build_offset_info: Vec<vk::AccelerationStructureBuildRangeInfoKHR>,
}

fn mesh_to_vk(device: &Device, mesh: &Mesh) -> BlasInput {
    let vertex_address_info = vk::BufferDeviceAddressInfo::builder().buffer(mesh.vertex_buffer.buffer);
    let vertex_address = unsafe { device.get_buffer_device_address(&vertex_address_info) };
    let index_address_info = vk::BufferDeviceAddressInfo::builder().buffer(mesh.index_buffer.buffer);
    let index_address = unsafe { device.get_buffer_device_address(&index_address_info) };

    let max_primitive_count = mesh.index_count() / 3;

    let triangles = vk::AccelerationStructureGeometryTrianglesDataKHR::builder()
        .vertex_format(vk::Format::R32G32B32_SFLOAT) // vec3 vertex position data
        .vertex_data(vk::DeviceOrHostAddressConstKHR { device_address: vertex_address })
        .vertex_stride(Vertex::stride())
        .index_type(vk::IndexType::UINT32)
        .index_data(vk::DeviceOrHostAddressConstKHR { device_address: index_address })
        // .transform_data()
        .max_vertex(mesh.vertex_count())
        .build();

    let as_geometry = vk::AccelerationStructureGeometryKHR::builder()
        .geometry_type(vk::GeometryTypeKHR::TRIANGLES)
        .geometry(vk::AccelerationStructureGeometryDataKHR { triangles })
        .build();

    let offset = vk::AccelerationStructureBuildRangeInfoKHR::builder()
        .first_vertex(0)
        .primitive_count(max_primitive_count)
        .primitive_offset(0)
        .transform_offset(0)
        .build();

    BlasInput {
        as_geometry: vec![as_geometry],
        as_build_offset_info: vec![offset],
    }
}

fn create_bottom_level_acceleration_structures(device: &Device, acceleration_structure_loader: &AccelerationStructure, meshes: &[Mesh]) {
    let blas_input = meshes.iter().map(|mesh|
        mesh_to_vk(device, mesh)
    ).collect::<Vec<_>>();

    let build_infos = blas_input.iter().map(|input| {
        vk::AccelerationStructureBuildGeometryInfoKHR::builder()
            .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
            .geometries(input.as_geometry.as_slice())
            // .geometries_ptrs(input.as_geometry.)
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
            .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
            .src_acceleration_structure(vk::AccelerationStructureKHR::null())
            .build()
    }).collect::<Vec<_>>();

    let max_primitive_counts = blas_input.iter().map(|input| {
        input.as_build_offset_info.iter().map(|info| info.primitive_count).max().unwrap()
    }).max().unwrap();

    // acceleration_structure_loader.get_acceleration_structure_build_sizes()
}