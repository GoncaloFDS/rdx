use ash::extensions::khr::Surface;
use ash::vk;

pub struct SwapchainImage {
    pub image_view: vk::ImageView,
    pub fence: vk::Fence,
    pub command_buffer: vk::CommandBuffer,
    pub framebuffer: vk::Framebuffer,
}

pub struct SwapchainConfig {
    pub format: vk::Format,
    pub extent: vk::Extent2D,
    pub frames_in_flight: u32,
}

impl SwapchainConfig {
    pub fn new(
        physical_device: vk::PhysicalDevice,
        surface_loader: &Surface,
        surface: vk::SurfaceKHR,
    ) -> Self {
        let caps = unsafe {
            surface_loader
                .get_physical_device_surface_capabilities(physical_device, surface)
                .unwrap()
        };
        let supported_formats = unsafe {
            surface_loader
                .get_physical_device_surface_formats(physical_device, surface)
                .unwrap()
        };
        let _supported_present_mode = unsafe {
            surface_loader
                .get_physical_device_surface_present_modes(physical_device, surface)
                .unwrap()
        };

        let format = supported_formats[0].format;
        let extent = caps.current_extent;

        SwapchainConfig {
            format,
            extent,
            frames_in_flight: 3,
        }
    }
}
