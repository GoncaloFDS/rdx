use crate::device::Device;
use crate::physical_device::PhysicalDeviceInfo;
use crate::resources::{Image, Semaphore};
use crate::surface::Surface;
use erupt::vk;

pub struct SwapchainImage {
    info: SwapchainImageInfo,
    handle: vk::SwapchainKHR,
}

pub struct SwapchainImageInfo {
    pub image: Image,
    pub wait: Semaphore,
    pub signal: Semaphore,
}

struct SwapchainImageAndSemaphores {
    image: Image,
    acquire: [Semaphore; 3],
    acquire_index: usize,
    release: [Semaphore; 3],
    release_index: usize,
}

struct SwapchainInner {
    handle: vk::SwapchainKHR,
    index: usize,
    images: Vec<SwapchainImageAndSemaphores>,
    format: vk::Format,
    extent: vk::Extent2D,
    usage: vk::ImageUsageFlags,
}

pub struct Swapchain {
    inner: Option<SwapchainInner>,
    retired: Vec<SwapchainInner>,
    retired_offset: u64,
    free_semaphore: Semaphore,
    surface: Surface,
}

impl Swapchain {
    pub fn new(device: &Device, surface: &Surface) -> Self {
        Swapchain {
            inner: None,
            retired: vec![],
            retired_offset: 0,
            free_semaphore: device.create_semaphore(),
            surface: surface.clone(),
        }
    }

    pub fn configure(&mut self, device: &Device, info: &PhysicalDeviceInfo) {
        let old_swapchain = if let Some(inner) = self.inner.take() {
            let handle = inner.handle;
            self.retired.push(inner);
            handle
        } else {
            vk::SwapchainKHR::null()
        };

        let swapchain = unsafe {
            device
                .handle()
                .create_swapchain_khr(
                    &vk::SwapchainCreateInfoKHRBuilder::new()
                        .surface(self.surface.handle())
                        .min_image_count(
                            3.min(info.surface_capabilities.max_image_count)
                                .max(info.surface_capabilities.min_image_count),
                        )
                        .image_format(info.surface_format.format)
                        .image_color_space(info.surface_format.color_space)
                        .image_extent(info.surface_capabilities.current_extent)
                        .image_array_layers(1)
                        .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
                        .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
                        .pre_transform(info.surface_capabilities.current_transform)
                        .composite_alpha(vk::CompositeAlphaFlagBitsKHR::OPAQUE_KHR)
                        .present_mode(info.present_mode)
                        .clipped(true)
                        .old_swapchain(old_swapchain),
                    None,
                )
                .unwrap()
        };

        device.swapchains().lock().insert(swapchain);

        let images = unsafe {
            device
                .handle()
                .get_swapchain_images_khr(swapchain, None)
                .unwrap()
        };

        let semaphores = images
            .into_iter()
            .map(|_| {
                (
                    [
                        device.create_semaphore(),
                        device.create_semaphore(),
                        device.create_semaphore(),
                    ],
                    [
                        device.create_semaphore(),
                        device.create_semaphore(),
                        device.create_semaphore(),
                    ],
                )
            })
            .collect::<Vec<_>>();
    }
}
