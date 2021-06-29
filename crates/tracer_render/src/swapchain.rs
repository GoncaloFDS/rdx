use crate::device::Device;
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
        let handle = surface.handle();
        let instance = device.instance();

        todo!()
    }
}
