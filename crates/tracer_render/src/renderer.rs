use crate::device::Device;
use crate::physical_device::{PhysicalDevice, VALIDATION_LAYER};
use crate::queue::Queue;
use crate::render_context::RenderContext;
use crate::surface::Surface;
use erupt::utils::surface;
use erupt::{
    cstr, vk, DeviceLoader, EntryLoader, ExtendableFromConst, ExtendableFromMut, InstanceLoader,
};
use std::ffi::{c_void, CStr, CString};
use std::os::raw::c_char;
use std::sync::Arc;
use winit::window::Window;

const FRAMES_IN_FLIGHT: usize = 2;

pub struct Renderer {
    debug_messenger: vk::DebugUtilsMessengerEXT,
    render_context: RenderContext,
    instance: Arc<InstanceLoader>,
    entry: EntryLoader,
}

impl Renderer {
    pub fn new(window: &Window) -> Self {
        let entry = EntryLoader::new().unwrap();

        let instance = Arc::new(Renderer::create_instance(window, &entry));

        let debug_messenger = Renderer::create_debug_messenger(&instance);

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
            debug_messenger,
            render_context,
            instance,
            entry,
        }
    }

    fn create_instance(window: &Window, entry: &EntryLoader) -> InstanceLoader {
        let app_info =
            vk::ApplicationInfoBuilder::new().api_version(vk::make_api_version(1, 2, 0, 0));

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

        unsafe { InstanceLoader::new(&entry, &instance_info, None).unwrap() }
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

            unsafe { instance.create_debug_utils_messenger_ext(&messenger_info, None) }.unwrap()
        } else {
            Default::default()
        }
    }
}

impl Drop for Renderer {
    fn drop(&mut self) {
        unsafe {
            if !self.debug_messenger.is_null() {
                self.instance
                    .destroy_debug_utils_messenger_ext(Some(self.debug_messenger), None);
            }

            self.instance.destroy_instance(None);
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
    unsafe { device.create_shader_module(&module_info, None) }.unwrap()
}
