use ash::version::{EntryV1_0, InstanceV1_0};
use ash::vk;

#[derive(Debug, Clone)]
pub struct DeviceInfo {
    pub supported_extensions: Vec<vk::ExtensionProperties>,
    pub physical_device_properties: vk::PhysicalDeviceProperties,
    pub memory_properties: vk::PhysicalDeviceMemoryProperties,
    pub features: vk::PhysicalDeviceFeatures,
}

impl DeviceInfo {
    pub fn new(
        entry: &ash::Entry,
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
    ) -> Self {
        let supported_extensions = entry.enumerate_instance_extension_properties().unwrap();
        let physical_device_properties =
            unsafe { instance.get_physical_device_properties(physical_device) };
        let memory_properties =
            unsafe { instance.get_physical_device_memory_properties(physical_device) };
        let features = unsafe { instance.get_physical_device_features(physical_device) };

        DeviceInfo {
            supported_extensions,
            physical_device_properties,
            memory_properties,
            features,
        }
    }
}
