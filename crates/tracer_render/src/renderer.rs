use std::collections::HashSet;
use std::ffi::{c_void, CStr, CString};
use std::sync::Arc;

use ash::{Device, Entry, Instance, vk};
use ash::extensions::ext::DebugUtils;
use ash::extensions::khr::{AccelerationStructure, DeferredHostOperations, RayTracingPipeline, Surface, Swapchain};
use ash::version::{DeviceV1_0, EntryV1_0, InstanceV1_0};
use ash::vk::{BufferDeviceAddressInfo, BufferDeviceAddressInfoKHR, ExtBufferDeviceAddressFn, KhrBufferDeviceAddressFn, KhrDedicatedAllocationFn, KhrGetMemoryRequirements2Fn, KhrMaintenance3Fn, KhrPipelineLibraryFn, PhysicalDeviceBufferDeviceAddressFeatures, PhysicalDeviceBufferDeviceAddressFeaturesEXT, PhysicalDeviceBufferDeviceAddressFeaturesKHR, PhysicalDeviceFeatures};
use bevy::log::*;
use winit::window::Window;

use crate::device_info::DeviceInfo;
use crate::render_context::RenderContext;
use crate::vk_types::vk_to_string;

#[cfg(debug_assertions)]
const ENABLE_VALIDATION_LAYERS: bool = true;
#[cfg(not(debug_assertions))]
const ENABLE_VALIDATION_LAYERS: bool = false;

#[cfg(debug_assertions)]
const VALIDATION: &[&str] = &["VK_LAYER_KHRONOS_validation"];
#[cfg(not(debug_assertions))]
const VALIDATION: &[&str] = &[];

const DEVICE_EXTENSIONS: &[&str] = &[
    "VK_KHR_swapchain",
];

pub struct VulkanContext {
    pub entry: ash::Entry,
    pub instance: ash::Instance,
    pub device: ash::Device,
    pub physical_device: vk::PhysicalDevice,
}

pub struct Renderer {
    pub vk_context: Arc<VulkanContext>,

    pub surface_loader: Surface,
    pub surface: vk::SurfaceKHR,

    pub graphics_queue: vk::Queue,
    pub graphics_queue_family: u32,
    pub device_info: DeviceInfo,

    #[cfg(debug_assertions)]
    debug_utils_loader: DebugUtils,
    #[cfg(debug_assertions)]
    debug_callback: vk::DebugUtilsMessengerEXT,
}

impl Renderer {
    pub fn new(window: &Window) -> Self {
        let entry = unsafe { ash::Entry::new().unwrap() };

        let instance = create_vulkan_instance("tracer", &window, &entry);

        #[cfg(debug_assertions)]
            let (debug_utils_loader, debug_callback) = create_debug_messenger(&entry, &instance);

        let surface_loader = Surface::new(&entry, &instance);
        let surface =
            unsafe { ash_window::create_surface(&entry, &instance, window, None).unwrap() };

        let (physical_device, graphics_queue_family) =
            pick_physical_device(&instance, &surface_loader, surface);
        let device = create_logical_device(&instance, physical_device, graphics_queue_family);
        let device_info = DeviceInfo::new(&entry, &instance, physical_device);

        let graphics_queue = unsafe { device.get_device_queue(graphics_queue_family, 0) };

        let vk_context = VulkanContext {
            entry,
            instance,
            device,
            physical_device,
        };
        Renderer {
            vk_context: Arc::new(vk_context),
            surface_loader,
            surface,
            graphics_queue,
            graphics_queue_family,
            device_info,
            debug_utils_loader,
            debug_callback,
        }
    }

    pub fn draw_frame(&mut self, render_context: &mut RenderContext) {
        unsafe { render_context.draw(); }
    }
}

fn create_vulkan_instance(title: &str, window: &Window, entry: &Entry) -> Instance {
    let app_name = CString::new(title).unwrap();
    let engine_name = CString::new("Vulkan Engine").unwrap();
    let app_info = vk::ApplicationInfo::builder()
        .api_version(vk::make_version(1, 2, 0))
        .application_version(vk::make_version(0, 1, 0))
        .application_name(&app_name)
        .engine_version(vk::make_version(0, 1, 0))
        .engine_name(&engine_name);

    let mut extension_names = ash_window::enumerate_required_extensions(window).unwrap();
    if ENABLE_VALIDATION_LAYERS {
        extension_names.push(DebugUtils::name())
    }

    let extension_names = extension_names
        .iter()
        .map(|name| name.as_ptr())
        .collect::<Vec<_>>();

    let enabled_layer_names = VALIDATION
        .iter()
        .map(|layer_name| CString::new(*layer_name).unwrap())
        .collect::<Vec<_>>();
    let enabled_layer_names = enabled_layer_names
        .iter()
        .map(|layer_name| layer_name.as_ptr())
        .collect::<Vec<_>>();

    let create_info = vk::InstanceCreateInfo::builder()
        .application_info(&app_info)
        .enabled_extension_names(&extension_names)
        .enabled_layer_names(&enabled_layer_names);

    unsafe { entry.create_instance(&create_info, None).unwrap() }
}

#[cfg(debug_assertions)]
fn create_debug_messenger(
    entry: &Entry,
    instance: &Instance,
) -> (DebugUtils, vk::DebugUtilsMessengerEXT) {
    use std::ffi::c_void;
    unsafe extern "system" fn callback(
        message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
        message_type: vk::DebugUtilsMessageTypeFlagsEXT,
        p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
        _p_user_date: *mut c_void,
    ) -> vk::Bool32 {
        let types = match message_type {
            vk::DebugUtilsMessageTypeFlagsEXT::GENERAL => "[General]",
            vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE => "[Performance]",
            vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION => "[Validation]",
            _ => "[Unknown]",
        };
        let message = CStr::from_ptr((*p_callback_data).p_message);

        match message_severity {
            vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE => {
                trace!("{} {:?}", types, message)
            }
            vk::DebugUtilsMessageSeverityFlagsEXT::INFO => {
                info!("[Vulkan Validation] {} {:?}", types, message)
            }
            vk::DebugUtilsMessageSeverityFlagsEXT::WARNING => {
                warn!("[Vulkan Validation] {} {:?}", types, message)
            }
            vk::DebugUtilsMessageSeverityFlagsEXT::ERROR => {
                error!("[Vulkan Validation] {} {:?}", types, message)
            }
            _ => warn!("[Vulkan Validation] {} {:?}", types, message),
        };

        vk::FALSE
    }

    let debug_utils_messenger_create_info = vk::DebugUtilsMessengerCreateInfoEXT::builder()
        .message_severity(
            vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                // | vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE
                // | vk::DebugUtilsMessageSeverityFlagsEXT::INFO
                | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
        )
        .message_type(
            vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE
                | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION,
        )
        .pfn_user_callback(Some(callback));
    let debug_utils_loader = DebugUtils::new(entry, instance);
    let debug_callback = unsafe {
        debug_utils_loader
            .create_debug_utils_messenger(&debug_utils_messenger_create_info, None)
            .unwrap()
    };
    (debug_utils_loader, debug_callback)
}

fn pick_physical_device(
    instance: &Instance,
    surface_loader: &Surface,
    surface: vk::SurfaceKHR,
) -> (vk::PhysicalDevice, u32) {
    let physical_devices = unsafe { instance.enumerate_physical_devices().unwrap() };
    let mut graphics_queue_index = None;
    let physical_device = physical_devices.iter().find(|&&physical_device| {
        let queue_families =
            unsafe { instance.get_physical_device_queue_family_properties(physical_device) };

        queue_families
            .iter()
            .enumerate()
            .find(|(i, &queue_family)| {
                graphics_queue_index = Some(*i as u32);
                queue_family.queue_count > 0
                    && queue_family.queue_flags.contains(vk::QueueFlags::GRAPHICS)
            });

        let device_supports_surface = unsafe {
            surface_loader
                .get_physical_device_surface_support(
                    physical_device,
                    graphics_queue_index.unwrap(),
                    surface,
                )
                .unwrap()
        };

        let are_queue_families_supported = graphics_queue_index.is_some();

        // Check if the extensions specified in DEVICE_EXTENSIONS are supported.
        let is_device_extension_supported = {
            let available_extensions = unsafe {
                instance
                    .enumerate_device_extension_properties(physical_device)
                    .unwrap()
            };
            let available_extension_names = available_extensions
                .iter()
                .map(|extension| vk_to_string(&extension.extension_name))
                .collect::<Vec<_>>();

            DEVICE_EXTENSIONS.iter().all(|required_extension| {
                available_extension_names.contains(&required_extension.to_string())
            })
        };

        // Check if Swapchain is supported.
        let is_swapchain_supported = if is_device_extension_supported {
            let formats = unsafe {
                surface_loader
                    .get_physical_device_surface_formats(physical_device, surface)
                    .unwrap()
            };
            let present_modes = unsafe {
                surface_loader
                    .get_physical_device_surface_present_modes(physical_device, surface)
                    .unwrap()
            };
            !formats.is_empty() && !present_modes.is_empty()
        } else {
            false
        };

        are_queue_families_supported
            && is_device_extension_supported
            && is_swapchain_supported
            && device_supports_surface
    });

    (
        *physical_device.expect("Failed to select a valid physical device"),
        graphics_queue_index.unwrap(),
    )
}

fn create_logical_device(
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
    graphics_queue_index: u32,
) -> Device {
    let mut unique_queue_families = HashSet::new();
    unique_queue_families.insert(graphics_queue_index);

    let queue_priorities = [1.0];
    let mut queue_create_infos = vec![];
    for &queue_family in unique_queue_families.iter() {
        let queue_create_info = vk::DeviceQueueCreateInfo::builder()
            .queue_family_index(queue_family)
            .queue_priorities(&queue_priorities)
            .build();
        queue_create_infos.push(queue_create_info);
    }

    let enabled_extension_names = [
        Swapchain::name().as_ptr(),
        KhrDedicatedAllocationFn::name().as_ptr(),
        KhrGetMemoryRequirements2Fn::name().as_ptr(),
        AccelerationStructure::name().as_ptr(),
        RayTracingPipeline::name().as_ptr(),
        KhrMaintenance3Fn::name().as_ptr(),
        KhrPipelineLibraryFn::name().as_ptr(),
        DeferredHostOperations::name().as_ptr(),
    ];

    let mut features = vk::PhysicalDeviceVulkan12Features::builder()
        .buffer_device_address(true);

    let device_create_info = vk::DeviceCreateInfo::builder()
        .queue_create_infos(queue_create_infos.as_slice())
        .enabled_extension_names(&enabled_extension_names)
        .push_next(&mut features);


    unsafe {
        instance
            .create_device(physical_device, &device_create_info, None)
            .unwrap()
    }
}

impl Drop for Renderer {
    fn drop(&mut self) {
        unsafe {
            self.surface_loader.destroy_surface(self.surface, None);
            self.vk_context.device.destroy_device(None);

            #[cfg(debug_assertions)]
                self.debug_utils_loader
                .destroy_debug_utils_messenger(self.debug_callback, None);

            self.vk_context.instance.destroy_instance(None);
        }
    }
}
