use crate::device::Device;
use crate::encoder::Encoder;
use crate::swapchain::SwapchainImage;
use erupt::vk;
use erupt::vk::PresentInfoKHRBuilder;

pub struct Queue {
    handle: vk::Queue,
    pool: vk::CommandPool,
    device: Device,
}

impl Queue {
    pub fn new(handle: vk::Queue, device: Device) -> Self {
        Queue {
            handle,
            pool: vk::CommandPool::null(),
            device,
        }
    }

    pub fn create_enconder(&mut self) -> Encoder<'static> {
        todo!()
    }

    pub fn submit(&self) {
        todo!()
    }

    pub fn present(&mut self, image: SwapchainImage) {
        unsafe {
            self.device
                .handle()
                .queue_present_khr(
                    self.handle,
                    &PresentInfoKHRBuilder::new()
                        .wait_semaphores(&[image.info().signal.handle()])
                        .swapchains(&[image.handle()])
                        .image_indices(&[image.index()]),
                )
                .unwrap();
        }
    }
}

pub struct QueueId {
    pub family: usize,
    pub index: usize,
}
