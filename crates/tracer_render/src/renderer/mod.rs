use crate::debug::DebugMessenger;
use crate::instance;
use crate::physical_device::PhysicalDevice;
use crate::render_context::RenderContext;
use crate::surface::Surface;
use crate::swapchain::Swapchain;
use erupt::{vk, EntryLoader, InstanceLoader};
use std::sync::Arc;
use winit::window::Window;

pub use self::pass::*;
use crate::pipeline::{Pipeline, RasterPipeline};
use parking_lot::Mutex;

mod pass;

pub struct Renderer {
    surface: Surface,
    swapchain: Swapchain,
    debug_messenger: DebugMessenger,
    physical_device: PhysicalDevice,
    render_context: RenderContext,
    pipeline: RasterPipeline,
    instance: Arc<InstanceLoader>,
    entry: EntryLoader,
}

impl Renderer {
    pub fn new(window: &Window) -> Self {
        let entry = EntryLoader::new().unwrap();
        let instance = Arc::new(instance::create_instance(window, &entry));
        let debug_messenger = DebugMessenger::new(&instance);
        let surface = Surface::new(&instance, window);

        let device_extensions = vec![
            vk::KHR_SWAPCHAIN_EXTENSION_NAME,
            vk::KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
            vk::KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
            vk::KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME,
            vk::KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
        ];
        let physical_device = PhysicalDevice::select_one(&instance, &surface, &device_extensions);
        let (device, queue) = physical_device.create_device(instance.clone(), &device_extensions);
        let mut render_context = RenderContext::new(device, queue);

        let mut swapchain = render_context.create_swapchain(&surface);
        swapchain.configure(&render_context.device, physical_device.info());

        let pipeline = RasterPipeline::new(
            &render_context,
            physical_device.info().surface_format.format,
            physical_device.info().surface_capabilities.current_extent,
        );

        Renderer {
            surface,
            swapchain,
            debug_messenger,
            physical_device,
            render_context,
            pipeline,
            instance,
            entry,
        }
    }

    pub fn draw(&mut self) {
        let swapchain_image = loop {
            if let Some(swapchain_image) = self
                .swapchain
                .acquire_next_image(&self.render_context.device)
            {
                break swapchain_image;
            }
            self.swapchain
                .configure(&self.render_context.device, self.physical_device.info());
        };

        self.pipeline.draw(
            swapchain_image.info().image.clone(),
            &swapchain_image.info().wait,
            &swapchain_image.info().signal,
            &mut self.render_context,
        );

        self.render_context.queue.present(swapchain_image);
    }
}

impl Drop for Renderer {
    fn drop(&mut self) {
        unsafe {
            self.render_context.destroy_context();
            self.instance
                .destroy_surface_khr(Some(self.surface.handle()), None);
            self.debug_messenger.destroy(&self.instance);
            self.instance.destroy_instance(None);
        }
    }
}
