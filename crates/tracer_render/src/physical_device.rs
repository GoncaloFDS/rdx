use crate::debug::VALIDATION_LAYER;
use crate::device::Device;
use crate::queue::Queue;
use crate::surface::Surface;
use erupt::{vk, DeviceLoader, ExtendableFromConst, ExtendableFromMut, InstanceLoader};
use std::ffi::CStr;
use std::sync::Arc;

pub struct PhysicalDevice {
    info: PhysicalDeviceInfo,
    handle: vk::PhysicalDevice,
}

pub struct PhysicalDeviceInfo {
    pub queue_index: u32,
    pub surface_format: vk::SurfaceFormatKHR,
    pub present_mode: vk::PresentModeKHR,
    pub device_properties: vk::PhysicalDeviceProperties,
    pub surface_capabilities: vk::SurfaceCapabilitiesKHR,
    pub raytracing_properties: vk::PhysicalDeviceRayTracingPipelinePropertiesKHR,
    pub accel_properties: vk::PhysicalDeviceAccelerationStructurePropertiesKHR,
}

unsafe impl Send for PhysicalDeviceInfo {}
unsafe impl Sync for PhysicalDeviceInfo {}

impl PhysicalDevice {
    pub fn select_one(
        instance: &InstanceLoader,
        surface: &Surface,
        device_extensions: &[*const i8],
    ) -> Self {
        let devices = unsafe { instance.enumerate_physical_devices(None).unwrap() };

        devices
            .into_iter()
            .filter_map(|physical_device| {
                match PhysicalDevice::supports_requirements(
                    instance,
                    physical_device,
                    &surface,
                    device_extensions,
                ) {
                    None => None,
                    Some(info) => Some(PhysicalDevice {
                        info,
                        handle: physical_device,
                    }),
                }
            })
            .next()
            .unwrap_or_else(|| panic!("No supported devices found"))
    }

    fn supports_requirements(
        instance: &InstanceLoader,
        physical_device: vk::PhysicalDevice,
        surface: &Surface,
        device_extensions: &[*const i8],
    ) -> Option<PhysicalDeviceInfo> {
        let queue_family =
            unsafe { instance.get_physical_device_queue_family_properties(physical_device, None) };
        let queue_family =
            match queue_family
                .into_iter()
                .enumerate()
                .position(|(i, queue_family_properties)| {
                    let supports_surface = unsafe {
                        instance
                            .get_physical_device_surface_support_khr(
                                physical_device,
                                i as u32,
                                surface.handle(),
                            )
                            .unwrap()
                    };
                    queue_family_properties
                        .queue_flags
                        .contains(vk::QueueFlags::GRAPHICS)
                        && supports_surface
                }) {
                Some(queue_family) => queue_family as u32,
                None => return None,
            };

        let formats = unsafe {
            instance
                .get_physical_device_surface_formats_khr(physical_device, surface.handle(), None)
                .unwrap()
        };
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

        let present_mode = unsafe {
            instance.get_physical_device_surface_present_modes_khr(
                physical_device,
                surface.handle(),
                None,
            )
        };
        let present_mode = present_mode
            .unwrap()
            .into_iter()
            .find(|present_mode| present_mode == &vk::PresentModeKHR::MAILBOX_KHR)
            .unwrap_or(vk::PresentModeKHR::FIFO_KHR);

        let supported_device_extensions = unsafe {
            instance
                .enumerate_device_extension_properties(physical_device, None, None)
                .unwrap()
        };
        let device_extensions_supported = device_extensions.iter().all(|device_extension| {
            let device_extension = unsafe { CStr::from_ptr(*device_extension) };

            supported_device_extensions.iter().any(|properties| unsafe {
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

        let device_properties2 = unsafe {
            instance.get_physical_device_properties2(physical_device, Some(*properties2))
        };
        let device_properties = device_properties2.properties;

        let surface_capabilities = unsafe {
            instance
                .get_physical_device_surface_capabilities_khr(physical_device, surface.handle())
                .unwrap()
        };

        Some(PhysicalDeviceInfo {
            queue_index: queue_family,
            surface_format,
            present_mode,
            device_properties,
            surface_capabilities,
            accel_properties,
            raytracing_properties,
        })
    }

    pub fn info(&self) -> &PhysicalDeviceInfo {
        &self.info
    }

    pub fn create_device(
        &self,
        instance: Arc<InstanceLoader>,
        device_extensions: &[*const i8],
    ) -> (Device, Queue) {
        let queue_info = [vk::DeviceQueueCreateInfoBuilder::new()
            .queue_family_index(self.info.queue_index)
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

        let device =
            unsafe { DeviceLoader::new(&instance, self.handle, &device_info, None).unwrap() };
        let device = Device::new(instance.clone(), device, self.handle);

        let queue = unsafe { device.handle().get_device_queue(self.info.queue_index, 0) };
        let queue = Queue::new(queue, device.clone());

        (device, queue)
    }
}
