use std::sync::Arc;

use ash::{Device, vk};
use ash::extensions::khr::{Surface, Swapchain};
use ash::version::DeviceV1_0;

use crate::renderer::VulkanContext;
use crate::swapchain::{SwapchainConfig, SwapchainImage};

struct Frame {
    image_available_semaphore: vk::Semaphore,
    render_finished_semaphore: vk::Semaphore,
    fence: vk::Fence,
}

impl Frame {
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
        Frame {
            image_available_semaphore,
            render_finished_semaphore,
            fence,
        }
    }
}

pub struct RenderContext {
    vk_context: Arc<VulkanContext>,
    surface: vk::SurfaceKHR,
    swapchain_loader: Swapchain,
    swapchain: vk::SwapchainKHR,
    frames_in_flight: Vec<Frame>,
    swapchain_images: Vec<SwapchainImage>,
    graphics_queue: vk::Queue,
    command_pool: vk::CommandPool,
    swapchain_config: SwapchainConfig,
    render_pass: vk::RenderPass,
    current_frame: usize,
}

impl RenderContext {
    pub fn new(
        vk_context: Arc<VulkanContext>,
        surface: vk::SurfaceKHR,
        swapchain_config: SwapchainConfig,
        graphics_queue: vk::Queue,
        graphics_queue_family: u32,
    ) -> Self {
        let instance = &vk_context.instance;
        let device = &vk_context.device;
        let swapchain_loader = Swapchain::new(instance, device);
        let swapchain = create_swapchain(&swapchain_loader, surface, &swapchain_config, None);
        let swapchain_images = unsafe { swapchain_loader.get_swapchain_images(swapchain).unwrap() };

        let command_pool = create_command_pool(device, graphics_queue_family);
        let command_buffers =
            create_command_buffers(device, &command_pool, swapchain_images.len() as u32);

        let swapchain_image_views =
            create_swapchain_image_views(device, &swapchain_images, &swapchain_config);
        let swapchain_images = swapchain_image_views
            .into_iter()
            .zip(command_buffers)
            .map(|(image_view, command_buffer)| SwapchainImage {
                image_view,
                fence: Default::default(),
                command_buffer,
                framebuffer: Default::default(),
            })
            .collect();

        let mut frames_in_flight = Vec::with_capacity(swapchain_config.frames_in_flight as usize);
        for _ in 0..swapchain_config.frames_in_flight {
            frames_in_flight.push(Frame::new(&device))
        }

        RenderContext {
            vk_context,
            surface,
            swapchain_loader,
            swapchain,
            frames_in_flight,
            swapchain_images,
            graphics_queue,
            command_pool,
            swapchain_config,
            render_pass: vk::RenderPass::null(),
            current_frame: 0,
        }
    }
}

fn create_swapchain(
    swapchain_loader: &Swapchain,
    surface: vk::SurfaceKHR,
    swapchain_config: &SwapchainConfig,
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

fn create_swapchain_image_views(
    device: &Device,
    swapchain_images: &[vk::Image],
    swapchain_config: &SwapchainConfig,
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

impl Drop for RenderContext {
    fn drop(&mut self) {
        unsafe {
            let device = &self.vk_context.device;
            for swapchain_image in self.swapchain_images.iter() {
                device.destroy_framebuffer(swapchain_image.framebuffer, None);
                device.destroy_image_view(swapchain_image.image_view, None);
            }

            for frame in self.frames_in_flight.iter() {
                device.destroy_fence(frame.fence, None);
                device.destroy_semaphore(frame.image_available_semaphore, None);
                device.destroy_semaphore(frame.render_finished_semaphore, None);
            }

            self.swapchain_loader.destroy_swapchain(self.swapchain, None);
            device.destroy_command_pool(self.command_pool, None);
        }
    }
}