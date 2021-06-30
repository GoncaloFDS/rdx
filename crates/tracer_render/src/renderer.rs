use crate::debug::DebugMessenger;
use crate::instance;
use crate::physical_device::PhysicalDevice;
use crate::render_context::RenderContext;
use crate::surface::Surface;
use erupt::{vk, EntryLoader, InstanceLoader};
use std::sync::Arc;
use winit::window::Window;

pub struct Renderer {
    surface: Surface,
    debug_messenger: DebugMessenger,
    render_context: RenderContext,
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
        let render_context = RenderContext::new(device, queue);

        Renderer {
            surface,
            debug_messenger,
            render_context,
            instance,
            entry,
        }
    }
}

impl Drop for Renderer {
    fn drop(&mut self) {
        unsafe {
            self.instance
                .destroy_surface_khr(Some(self.surface.handle()), None);
        }
        self.render_context.destroy_context();
        self.debug_messenger.destroy(&self.instance);

        unsafe {
            self.instance.destroy_instance(None);
        }
    }
}
