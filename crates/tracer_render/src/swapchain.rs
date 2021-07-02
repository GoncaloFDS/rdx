use crate::device::Device;
use crate::image::ImageInfo;
use crate::physical_device::PhysicalDeviceInfo;
use crate::resources::{Image, Semaphore};
use crate::surface::Surface;
use erupt::vk;

pub struct SwapchainImage {
    info: SwapchainImageInfo,
    handle: vk::SwapchainKHR,
    index: u32,
}

impl SwapchainImage {
    pub fn info(&self) -> &SwapchainImageInfo {
        &self.info
    }

    pub fn handle(&self) -> vk::SwapchainKHR {
        self.handle
    }

    pub fn index(&self) -> u32 {
        self.index
    }
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
    images: Vec<SwapchainImageAndSemaphores>,
    extent: vk::Extent2D,
    format: vk::Format,
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
                        .image_usage(
                            vk::ImageUsageFlags::COLOR_ATTACHMENT
                                | vk::ImageUsageFlags::TRANSFER_DST,
                        )
                        .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
                        .pre_transform(info.surface_capabilities.current_transform)
                        .composite_alpha(vk::CompositeAlphaFlagBitsKHR::OPAQUE_KHR)
                        .present_mode(info.present_mode)
                        .clipped(true)
                        .queue_family_indices(&[info.queue_index])
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

        let semaphores = (0..images.len())
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

        let images = images
            .into_iter()
            .zip(semaphores)
            .map(|(image, (acquire, release))| SwapchainImageAndSemaphores {
                image: Image::new(
                    ImageInfo {
                        extent: info.surface_capabilities.current_extent,
                        format: info.surface_format.format,
                        levels: 1,
                        layers: 1,
                        samples: vk::SampleCountFlags::_1,
                        usage: vk::ImageUsageFlags::COLOR_ATTACHMENT,
                    },
                    image,
                    None,
                ),
                acquire,
                acquire_index: 0,
                release,
                release_index: 0,
            })
            .collect();

        self.inner = Some(SwapchainInner {
            handle: swapchain,
            images,
            extent: info.surface_capabilities.current_extent.into(),
            format: info.surface_format.format,
            usage: vk::ImageUsageFlags::COLOR_ATTACHMENT,
        })
    }

    pub fn acquire_next_image(&mut self, device: &Device) -> Option<SwapchainImage> {
        if let Some(inner) = self.inner.as_mut() {
            let wait = self.free_semaphore.clone();

            let index = unsafe {
                device
                    .handle()
                    .acquire_next_image_khr(inner.handle, !0, Some(wait.handle()), None)
                    .unwrap()
            };

            let image_and_semaphores = &mut inner.images[index as usize];

            std::mem::swap(
                &mut image_and_semaphores.acquire[image_and_semaphores.acquire_index % 3],
                &mut self.free_semaphore,
            );

            image_and_semaphores.acquire_index += 1;

            let signal =
                image_and_semaphores.release[image_and_semaphores.release_index % 3].clone();

            image_and_semaphores.release_index += 1;

            Some(SwapchainImage {
                info: SwapchainImageInfo {
                    image: image_and_semaphores.image.clone(),
                    wait,
                    signal,
                },
                handle: inner.handle,
                index,
            })
        } else {
            None
        }
    }
}
