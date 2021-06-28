use crate::device::Device;
use erupt::utils::surface;
use erupt::vk1_0::{DeviceMemory, PhysicalDevice};
use erupt::{
    cstr, vk, DefaultEntryLoader, DeviceLoader, EntryLoader, ExtendableFrom, InstanceLoader,
};
use gpu_alloc::GpuAllocator;
use parking_lot::Mutex;
use std::ffi::{c_void, CStr, CString};
use std::os::raw::c_char;
use std::sync::Arc;
use winit::window::Window;

const VALIDATION_LAYER: *const c_char = cstr!("VK_LAYER_KHRONOS_validation");
const FRAMES_IN_FLIGHT: usize = 2;

#[derive(Copy, Clone)]
pub struct RendererProperties {
    pub queue_index: u32,
    pub surface_format: vk::SurfaceFormatKHR,
    pub present_mode: vk::PresentModeKHR,
    pub device_properties: vk::PhysicalDeviceProperties,
    pub surface_capabilities: vk::SurfaceCapabilitiesKHR,
    pub raytracing_properties: vk::PhysicalDeviceRayTracingPipelinePropertiesKHR,
    pub accel_properties: vk::PhysicalDeviceAccelerationStructurePropertiesKHR,
}

unsafe impl Send for RendererProperties {}
unsafe impl Sync for RendererProperties {}

pub struct Renderer {
    physical_device: vk::PhysicalDevice,
    queue: vk::Queue,

    renderer_properties: RendererProperties,

    debug_messenger: vk::DebugUtilsMessengerEXT,
    device: Device,
    instance: InstanceLoader,
    entry: DefaultEntryLoader,
}

impl Renderer {
    pub fn new(window: &Window) -> Self {
        let entry = erupt::EntryLoader::new().unwrap();
        let device_extensions = vec![
            vk::KHR_SWAPCHAIN_EXTENSION_NAME,
            vk::KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
            vk::KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
            vk::KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME,
            vk::KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
        ];

        let instance = Renderer::create_instance(window, &entry);

        let debug_messenger = Renderer::create_debug_messenger(&instance);

        let surface = unsafe { surface::create_surface(&instance, window, None).unwrap() };

        let (physical_device, renderer_properties) =
            { Renderer::pick_physical_device(&instance, surface, &device_extensions) };

        tracing::debug!("Using physical device: {:?}", unsafe {
            CStr::from_ptr(renderer_properties.device_properties.device_name.as_ptr())
        });

        let (device, queue) = Renderer::create_logical_device(
            &instance,
            &physical_device,
            renderer_properties.queue_index,
            &device_extensions,
        );
        let device = Device::new(&instance, device, physical_device);

        // let swapchain = Renderer::create_swapchain(&device, surface, &renderer_properties);
        //
        // let (swapchain_images, swapchain_image_views) =
        //     Renderer::get_swapchain_images(&device, swapchain, &renderer_properties);
        //
        // let render_pass = Renderer::create_default_render_pass(&device, &renderer_properties);
        //
        // let (graphics_pipeline, graphics_pipeline_layout) =
        //     Renderer::create_graphics_pipeline(&device, &renderer_properties, render_pass);
        //
        // let swapchain_framebuffers = Renderer::create_framebuffers(
        //     &device,
        //     render_pass,
        //     &swapchain_image_views,
        //     renderer_properties.surface_capabilities.current_extent,
        // );
        //
        // let command_pool = CommandPool::new(
        //     device.clone(),
        //     queue,
        //     renderer_properties.queue_index,
        //     vk::CommandPoolCreateFlags::TRANSIENT,
        // );
        //
        // let command_buffers = command_pool.create_command_buffers(
        //     vk::CommandBufferLevel::PRIMARY,
        //     swapchain_framebuffers.len() as u32,
        // );
        //
        // let (image_available_semaphores, render_finished_semaphores, fences) =
        //     Renderer::create_sync_objects(&device);
        //
        // let raytracing_context = RaytracingContext::new(
        //     device.clone(),
        //     allocator.clone(),
        //     renderer_properties,
        //     queue,
        // );

        Renderer {
            physical_device,
            queue,
            renderer_properties,
            debug_messenger,
            device,
            instance,
            entry,
        }
    }

    fn create_instance<T>(window: &Window, entry: &EntryLoader<T>) -> InstanceLoader {
        let app_info = vk::ApplicationInfoBuilder::new().api_version(vk::make_version(1, 2, 0));

        let mut instance_extensions = surface::enumerate_required_extensions(window).unwrap();
        if cfg!(debug_assertions) {
            instance_extensions.push(vk::EXT_DEBUG_UTILS_EXTENSION_NAME);
        }

        #[cfg(target_os = "windows")]
        {
            instance_extensions.push(vk::KHR_WIN32_SURFACE_EXTENSION_NAME);
        }

        let mut instance_layers = Vec::new();
        if cfg!(debug_assertions) {
            instance_layers.push(VALIDATION_LAYER);
        }

        let instance_info = vk::InstanceCreateInfoBuilder::new()
            .application_info(&app_info)
            .enabled_extension_names(&instance_extensions)
            .enabled_layer_names(&instance_layers);

        InstanceLoader::new(&entry, &instance_info, None).unwrap()
    }

    fn create_debug_messenger(instance: &InstanceLoader) -> vk::DebugUtilsMessengerEXT {
        unsafe extern "system" fn debug_callback(
            message_severity: vk::DebugUtilsMessageSeverityFlagBitsEXT,
            message_type: vk::DebugUtilsMessageTypeFlagsEXT,
            p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
            _p_user_data: *mut c_void,
        ) -> vk::Bool32 {
            let types = match message_type {
                vk::DebugUtilsMessageTypeFlagsEXT::GENERAL_EXT => "[General]",
                vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE_EXT => "[Performance]",
                vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION_EXT => "[Validation]",
                _ => "[Unknown]",
            };
            let message = CStr::from_ptr((*p_callback_data).p_message);

            match message_severity {
                vk::DebugUtilsMessageSeverityFlagBitsEXT::VERBOSE_EXT => {
                    tracing::trace!("{} {:?}", types, message)
                }
                vk::DebugUtilsMessageSeverityFlagBitsEXT::INFO_EXT => {
                    tracing::info!("{} {:?}", types, message)
                }
                vk::DebugUtilsMessageSeverityFlagBitsEXT::WARNING_EXT => {
                    tracing::warn!("{} {:?}", types, message)
                }
                vk::DebugUtilsMessageSeverityFlagBitsEXT::ERROR_EXT => {
                    tracing::error!("{} {:?}", types, message)
                }
                _ => tracing::warn!("{} {:?}", types, message),
            };

            vk::FALSE
        }
        if cfg!(debug_assertions) {
            let messenger_info = vk::DebugUtilsMessengerCreateInfoEXTBuilder::new()
                .message_severity(
                    vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE_EXT
                        | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING_EXT
                        | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR_EXT,
                )
                .message_type(
                    vk::DebugUtilsMessageTypeFlagsEXT::GENERAL_EXT
                        | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION_EXT
                        | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE_EXT,
                )
                .pfn_user_callback(Some(debug_callback));

            unsafe { instance.create_debug_utils_messenger_ext(&messenger_info, None, None) }
                .unwrap()
        } else {
            Default::default()
        }
    }

    fn pick_physical_device(
        instance: &InstanceLoader,
        surface: vk::SurfaceKHR,
        device_extensions: &[*const i8],
    ) -> (vk::PhysicalDevice, RendererProperties) {
        let physical_devices = unsafe { instance.enumerate_physical_devices(None) };
        let chosen = physical_devices
            .unwrap()
            .into_iter()
            .filter_map(|physical_device| unsafe {
                let queue_family = match instance
                    .get_physical_device_queue_family_properties(physical_device, None)
                    .into_iter()
                    .enumerate()
                    .position(|(i, queue_family_properties)| {
                        queue_family_properties
                            .queue_flags
                            .contains(vk::QueueFlags::GRAPHICS)
                            && instance
                                .get_physical_device_surface_support_khr(
                                    physical_device,
                                    i as u32,
                                    surface,
                                    None,
                                )
                                .unwrap()
                    }) {
                    Some(queue_family) => queue_family as u32,
                    None => return None,
                };

                let formats = instance
                    .get_physical_device_surface_formats_khr(physical_device, surface, None)
                    .unwrap();
                let surface_format = match formats
                    .iter()
                    .find(|surface_format| {
                        surface_format.format == vk::Format::B8G8R8A8_SRGB
                            && surface_format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR_KHR
                    })
                    .or_else(|| formats.get(0))
                {
                    Some(surface_format) => *surface_format,
                    None => return None,
                };

                let present_mode = instance
                    .get_physical_device_surface_present_modes_khr(physical_device, surface, None)
                    .unwrap()
                    .into_iter()
                    .find(|present_mode| present_mode == &vk::PresentModeKHR::MAILBOX_KHR)
                    .unwrap_or(vk::PresentModeKHR::FIFO_KHR);

                let supported_device_extensions = instance
                    .enumerate_device_extension_properties(physical_device, None, None)
                    .unwrap();
                let device_extensions_supported =
                    device_extensions.iter().all(|device_extension| {
                        let device_extension = CStr::from_ptr(*device_extension);

                        supported_device_extensions.iter().any(|properties| {
                            CStr::from_ptr(properties.extension_name.as_ptr()) == device_extension
                        })
                    });

                if !device_extensions_supported {
                    return None;
                }

                let mut accel_properties =
                    vk::PhysicalDeviceAccelerationStructurePropertiesKHRBuilder::new().build();
                let mut raytracing_properties =
                    vk::PhysicalDeviceRayTracingPipelinePropertiesKHRBuilder::new().build();
                let properties2 = vk::PhysicalDeviceProperties2Builder::new()
                    .extend_from(&mut accel_properties)
                    .extend_from(&mut raytracing_properties);

                let device_properties2 =
                    instance.get_physical_device_properties2(physical_device, Some(*properties2));
                let device_properties = device_properties2.properties;

                tracing::info!("ray tracing properties");
                tracing::info!(
                    " max_geometry_count: {}",
                    accel_properties.max_geometry_count
                );
                tracing::info!(
                    " shder_group_handle_size: {}",
                    raytracing_properties.shader_group_handle_size
                );

                let surface_capabilities = instance
                    .get_physical_device_surface_capabilities_khr(physical_device, surface, None)
                    .unwrap();

                let renderer_properties = RendererProperties {
                    queue_index: queue_family,
                    surface_format,
                    present_mode,
                    device_properties,
                    surface_capabilities,
                    accel_properties,
                    raytracing_properties,
                };
                Some((physical_device, renderer_properties))
            })
            .max_by_key(
                |(_, properties)| match properties.device_properties.device_type {
                    vk::PhysicalDeviceType::DISCRETE_GPU => 2,
                    vk::PhysicalDeviceType::INTEGRATED_GPU => 1,
                    _ => 0,
                },
            )
            .expect("No suitable physical device found");
        chosen
    }

    fn create_logical_device(
        instance: &InstanceLoader,
        physical_device: &vk::PhysicalDevice,
        queue_index: u32,
        device_extensions: &[*const i8],
    ) -> (DeviceLoader, vk::Queue) {
        let queue_info = [vk::DeviceQueueCreateInfoBuilder::new()
            .queue_family_index(queue_index)
            .queue_priorities(&[1.0])];
        let features = vk::PhysicalDeviceFeaturesBuilder::new();

        let mut device_layers = Vec::new();

        if cfg!(debug_assertions) {
            device_layers.push(VALIDATION_LAYER)
        }

        let mut buffer_device_address_features =
            vk::PhysicalDeviceBufferDeviceAddressFeaturesBuilder::new().buffer_device_address(true);
        let mut indexing_features = vk::PhysicalDeviceDescriptorIndexingFeaturesBuilder::new()
            .runtime_descriptor_array(true);
        let mut reset_query_features =
            vk::PhysicalDeviceHostQueryResetFeaturesBuilder::new().host_query_reset(true);
        let mut acceleration_structure_features =
            vk::PhysicalDeviceAccelerationStructureFeaturesKHRBuilder::new()
                .acceleration_structure(true);
        let mut ray_tracing_features =
            vk::PhysicalDeviceRayTracingPipelineFeaturesKHRBuilder::new()
                .ray_tracing_pipeline(true);

        let device_info = vk::DeviceCreateInfoBuilder::new()
            .queue_create_infos(&queue_info)
            .enabled_features(&features)
            .enabled_extension_names(&device_extensions)
            .enabled_layer_names(&device_layers)
            .extend_from(&mut buffer_device_address_features)
            .extend_from(&mut indexing_features)
            .extend_from(&mut reset_query_features)
            .extend_from(&mut acceleration_structure_features)
            .extend_from(&mut ray_tracing_features);

        let device = DeviceLoader::new(instance, *physical_device, &device_info, None).unwrap();
        let queue = unsafe { device.get_device_queue(queue_index, 0, None) };
        (device, queue)
    }

    fn create_gpu_allocator(
        instance: &InstanceLoader,
        physical_device: PhysicalDevice,
    ) -> GpuAllocator<DeviceMemory> {
        let config = gpu_alloc::Config::i_am_prototyping();
        let properties =
            unsafe { gpu_alloc_erupt::device_properties(&instance, physical_device).unwrap() };

        GpuAllocator::new(config, properties)
    }

    fn create_swapchain(
        device: &DeviceLoader,
        surface: vk::SurfaceKHR,
        presentation_settings: &RendererProperties,
    ) -> vk::SwapchainKHR {
        let surface_caps = presentation_settings.surface_capabilities;
        let mut image_count = surface_caps.min_image_count + 1;
        if surface_caps.max_image_count > 0 && image_count > surface_caps.max_image_count {
            image_count = surface_caps.max_image_count;
        }

        let swapchain_info = vk::SwapchainCreateInfoKHRBuilder::new()
            .surface(surface)
            .min_image_count(image_count)
            .image_format(presentation_settings.surface_format.format)
            .image_color_space(presentation_settings.surface_format.color_space)
            .image_extent(surface_caps.current_extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(surface_caps.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagBitsKHR::OPAQUE_KHR)
            .present_mode(presentation_settings.present_mode)
            .clipped(true)
            .old_swapchain(vk::SwapchainKHR::null());

        unsafe { device.create_swapchain_khr(&swapchain_info, None, None) }.unwrap()
    }

    fn get_swapchain_images(
        device: &DeviceLoader,
        swapchain: vk::SwapchainKHR,
        presentation_settings: &RendererProperties,
    ) -> (Vec<vk::Image>, Vec<vk::ImageView>) {
        let swapchain_images = unsafe { device.get_swapchain_images_khr(swapchain, None) }.unwrap();

        let swapchain_image_views: Vec<_> = swapchain_images
            .iter()
            .map(|swapchain_image| {
                let image_view_info = vk::ImageViewCreateInfoBuilder::new()
                    .image(*swapchain_image)
                    .view_type(vk::ImageViewType::_2D)
                    .format(presentation_settings.surface_format.format)
                    .components(vk::ComponentMapping {
                        r: vk::ComponentSwizzle::IDENTITY,
                        g: vk::ComponentSwizzle::IDENTITY,
                        b: vk::ComponentSwizzle::IDENTITY,
                        a: vk::ComponentSwizzle::IDENTITY,
                    })
                    .subresource_range(
                        vk::ImageSubresourceRangeBuilder::new()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .base_mip_level(0)
                            .level_count(1)
                            .base_array_layer(0)
                            .layer_count(1)
                            .build(),
                    );
                unsafe { device.create_image_view(&image_view_info, None, None) }.unwrap()
            })
            .collect();

        (swapchain_images, swapchain_image_views)
    }

    fn create_default_render_pass(
        device: &DeviceLoader,
        presentation_settings: &RendererProperties,
    ) -> vk::RenderPass {
        let attachments = vec![vk::AttachmentDescriptionBuilder::new()
            .format(presentation_settings.surface_format.format)
            .samples(vk::SampleCountFlagBits::_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::PRESENT_SRC_KHR)];

        let color_attachment_refs = vec![vk::AttachmentReferenceBuilder::new()
            .attachment(0)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)];
        let subpasses = vec![vk::SubpassDescriptionBuilder::new()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(&color_attachment_refs)];
        let dependencies = vec![vk::SubpassDependencyBuilder::new()
            .src_subpass(vk::SUBPASS_EXTERNAL)
            .dst_subpass(0)
            .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .src_access_mask(vk::AccessFlags::empty())
            .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)];

        let render_pass_info = vk::RenderPassCreateInfoBuilder::new()
            .attachments(&attachments)
            .subpasses(&subpasses)
            .dependencies(&dependencies);
        unsafe { device.create_render_pass(&render_pass_info, None, None) }.unwrap()
    }

    fn create_graphics_pipeline(
        device: &DeviceLoader,
        presentation_settings: &RendererProperties,
        render_pass: vk::RenderPass,
    ) -> (vk::Pipeline, vk::PipelineLayout) {
        let entry_point = CString::new("main").unwrap();
        let shader_vert = create_shader_module(&device, "assets/shaders/shader.vert.spv");
        let shader_frag = create_shader_module(&device, "assets/shaders/shader.frag.spv");

        let shader_stages = vec![
            vk::PipelineShaderStageCreateInfoBuilder::new()
                .stage(vk::ShaderStageFlagBits::VERTEX)
                .module(shader_vert)
                .name(&entry_point),
            vk::PipelineShaderStageCreateInfoBuilder::new()
                .stage(vk::ShaderStageFlagBits::FRAGMENT)
                .module(shader_frag)
                .name(&entry_point),
        ];

        let vertex_input = vk::PipelineVertexInputStateCreateInfoBuilder::new();

        let input_assembly = vk::PipelineInputAssemblyStateCreateInfoBuilder::new()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
            .primitive_restart_enable(false);

        let surface_capabilites = presentation_settings.surface_capabilities;
        let viewports = vec![vk::ViewportBuilder::new()
            .x(0.0)
            .y(0.0)
            .width(surface_capabilites.current_extent.width as f32)
            .height(surface_capabilites.current_extent.height as f32)
            .min_depth(0.0)
            .max_depth(1.0)];
        let scissors = vec![vk::Rect2DBuilder::new()
            .offset(vk::Offset2D { x: 0, y: 0 })
            .extent(surface_capabilites.current_extent)];
        let viewport_state = vk::PipelineViewportStateCreateInfoBuilder::new()
            .viewports(&viewports)
            .scissors(&scissors);

        let rasterizer = vk::PipelineRasterizationStateCreateInfoBuilder::new()
            .depth_clamp_enable(false)
            .rasterizer_discard_enable(false)
            .polygon_mode(vk::PolygonMode::FILL)
            .line_width(1.0)
            .cull_mode(vk::CullModeFlags::BACK)
            .front_face(vk::FrontFace::CLOCKWISE)
            .depth_clamp_enable(false);

        let multisampling = vk::PipelineMultisampleStateCreateInfoBuilder::new()
            .sample_shading_enable(false)
            .rasterization_samples(vk::SampleCountFlagBits::_1);

        let color_blend_attachments = vec![vk::PipelineColorBlendAttachmentStateBuilder::new()
            .color_write_mask(
                vk::ColorComponentFlags::R
                    | vk::ColorComponentFlags::G
                    | vk::ColorComponentFlags::B
                    | vk::ColorComponentFlags::A,
            )
            .blend_enable(false)];
        let color_blending = vk::PipelineColorBlendStateCreateInfoBuilder::new()
            .logic_op_enable(false)
            .attachments(&color_blend_attachments);

        let pipeline_layout_info = vk::PipelineLayoutCreateInfoBuilder::new();
        let pipeline_layout =
            unsafe { device.create_pipeline_layout(&pipeline_layout_info, None, None) }.unwrap();

        let pipeline_info = vk::GraphicsPipelineCreateInfoBuilder::new()
            .stages(&shader_stages)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterizer)
            .multisample_state(&multisampling)
            .color_blend_state(&color_blending)
            .layout(pipeline_layout)
            .render_pass(render_pass)
            .subpass(0);

        let pipeline =
            unsafe { device.create_graphics_pipelines(None, &[pipeline_info], None) }.unwrap()[0];

        unsafe { device.destroy_shader_module(Some(shader_frag), None) }
        unsafe { device.destroy_shader_module(Some(shader_vert), None) }

        (pipeline, pipeline_layout)
    }

    fn create_framebuffers(
        device: &DeviceLoader,
        render_pass: vk::RenderPass,
        swapchain_image_views: &[vk::ImageView],
        extent: vk::Extent2D,
    ) -> Vec<vk::Framebuffer> {
        swapchain_image_views
            .iter()
            .map(|image_view| {
                let attachments = vec![*image_view];
                let framebuffer_info = vk::FramebufferCreateInfoBuilder::new()
                    .render_pass(render_pass)
                    .attachments(&attachments)
                    .width(extent.width)
                    .height(extent.height)
                    .layers(1);

                unsafe { device.create_framebuffer(&framebuffer_info, None, None) }.unwrap()
            })
            .collect()
    }

    fn create_sync_objects(
        device: &DeviceLoader,
    ) -> (Vec<vk::Semaphore>, Vec<vk::Semaphore>, Vec<vk::Fence>) {
        let semaphore_info = vk::SemaphoreCreateInfoBuilder::new();
        let image_available_semaphores: Vec<_> = (0..FRAMES_IN_FLIGHT)
            .map(|_| unsafe { device.create_semaphore(&semaphore_info, None, None) }.unwrap())
            .collect();
        let render_finished_semaphores: Vec<_> = (0..FRAMES_IN_FLIGHT)
            .map(|_| unsafe { device.create_semaphore(&semaphore_info, None, None) }.unwrap())
            .collect();

        let fence_info = vk::FenceCreateInfoBuilder::new().flags(vk::FenceCreateFlags::SIGNALED);
        let in_flight_fences: Vec<_> = (0..FRAMES_IN_FLIGHT)
            .map(|_| unsafe { device.create_fence(&fence_info, None, None) }.unwrap())
            .collect();

        (
            image_available_semaphores,
            render_finished_semaphores,
            in_flight_fences,
        )
    }

    pub fn init(&mut self) {
        // self.raytracing_context.create_offscreen_render();
        // self.raytracing_context.create_bottom_level_as();
        // self.raytracing_context.create_top_level_as();
        // self.raytracing_context.create_descriptor_set();
        // self.raytracing_context.create_raytracing_pipeline();
        // self.raytracing_context.create_shader_binding_table();
    }
}

impl Drop for Renderer {
    fn drop(&mut self) {
        unsafe {
            // self.device.device_wait_idle().unwrap();
            //
            // for &semaphore in self
            //     .image_available_semaphores
            //     .iter()
            //     .chain(self.render_finished_semaphores.iter())
            // {
            //     self.device.destroy_semaphore(Some(semaphore), None);
            // }
            //
            // for &fence in &self.fences {
            //     self.device.destroy_fence(Some(fence), None);
            // }
            //
            // self.command_pool.destroy();
            //
            // for &framebuffer in &self.swapchain_framebuffers {
            //     self.device.destroy_framebuffer(Some(framebuffer), None);
            // }
            //
            // self.device
            //     .destroy_pipeline(Some(self.graphics_pipeline), None);
            //
            // self.device
            //     .destroy_render_pass(Some(self.render_pass), None);
            //
            // self.device
            //     .destroy_pipeline_layout(Some(self.graphics_pipeline_layout), None);
            //
            // for &image_view in &self.swapchain_image_views {
            //     self.device.destroy_image_view(Some(image_view), None);
            // }
            //
            // self.device
            //     .destroy_swapchain_khr(Some(self.swapchain), None);
            //
            // self.raytracing_context.destroy();
            //
            // self.device.destroy_device(None);
            //
            // self.instance.destroy_surface_khr(Some(self.surface), None);
            //
            // if !self.debug_messenger.is_null() {
            //     self.instance
            //         .destroy_debug_utils_messenger_ext(Some(self.debug_messenger), None);
            // }
            //
            // self.instance.destroy_instance(None);
        }
    }
}

pub fn create_shader_module(device: &DeviceLoader, file: &str) -> vk::ShaderModule {
    use std::fs::File;
    use std::io::*;
    let mut shader_file = File::open(file).unwrap_or_else(|_| panic!("Failed to open {}", file));
    let mut bytes = Vec::new();
    shader_file.read_to_end(&mut bytes).unwrap();
    let spv = erupt::utils::decode_spv(&bytes).unwrap();
    let module_info = vk::ShaderModuleCreateInfoBuilder::new().code(&spv);
    unsafe { device.create_shader_module(&module_info, None, None) }.unwrap()
}
