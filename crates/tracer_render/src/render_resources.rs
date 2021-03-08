use ash::{Device, Instance, vk};
use vk_mem::Allocator;

use crate::render_context::RenderContext;
use crate::renderer::Renderer;
use crate::swapchain::SwapchainConfig;

pub struct RenderResources {
    pub allocator: vk_mem::Allocator,
    pub render_context: RenderContext,
}

impl RenderResources {
    pub fn new(renderer: &Renderer) -> Self {
        let allocator = create_vulkan_allocator(
            &renderer.vk_context.device,
            &renderer.vk_context.instance,
            renderer.physical_device,
        );

        let swapchain_config = SwapchainConfig::new(
            renderer.physical_device,
            &renderer.surface_loader,
            renderer.surface,
        );

        let render_context = RenderContext::new(
            renderer.vk_context.clone(),
            renderer.surface,
            swapchain_config,
            renderer.graphics_queue,
            renderer.graphics_queue_family,
        );

        RenderResources {
            allocator,
            render_context,
        }
    }
}

fn create_vulkan_allocator(
    device: &Device,
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
) -> Allocator {
    let create_info = vk_mem::AllocatorCreateInfo {
        physical_device,
        device: device.clone(),
        instance: instance.clone(),
        flags: vk_mem::AllocatorCreateFlags::empty(),
        preferred_large_heap_block_size: 0,
        frame_in_use_count: 0,
        heap_size_limits: None,
    };
    vk_mem::Allocator::new(&create_info).unwrap()
}
