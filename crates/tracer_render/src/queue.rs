use crate::command_buffer::CommandBuffer;
use crate::device::Device;
use crate::encoder::Encoder;
use crate::resources::{Fence, Semaphore};
use crate::swapchain::SwapchainImage;
use erupt::vk;
use erupt::vk::{PipelineStageFlags, PresentInfoKHRBuilder};
use smallvec::SmallVec;

pub struct Queue {
    handle: vk::Queue,
    pool: vk::CommandPool,
    device: Device,
    family_index: u32,
}

impl Queue {
    pub fn new(handle: vk::Queue, device: Device, family_index: u32) -> Self {
        Queue {
            handle,
            pool: vk::CommandPool::null(),
            device,
            family_index,
        }
    }

    pub fn create_enconder(&mut self) -> Encoder<'static> {
        if self.pool.is_null() {
            self.pool = unsafe {
                self.device
                    .handle()
                    .create_command_pool(
                        &vk::CommandPoolCreateInfoBuilder::new()
                            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
                            .queue_family_index(self.family_index),
                        None,
                    )
                    .unwrap()
            }
        }

        let command_buffer = unsafe {
            self.device
                .handle()
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfoBuilder::new()
                        .command_pool(self.pool)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(1),
                )
                .unwrap()
                .remove(0)
        };

        let command_buffer = CommandBuffer::new(command_buffer, self.family_index);

        Encoder::new(command_buffer)
    }

    pub fn submit(
        &mut self,
        wait: &[(PipelineStageFlags, Semaphore)],
        command_buffer: CommandBuffer,
        signal: &[Semaphore],
        fence: Option<&Fence>,
    ) {
        let (wait_stages, wait_semaphores) = wait
            .iter()
            .map(|(stage, semaphore)| (*stage, semaphore.handle()))
            .unzip::<_, _, SmallVec<[_; 8]>, SmallVec<[_; 8]>>();

        let signal_semaphores = signal
            .iter()
            .map(|semaphore| semaphore.handle())
            .collect::<SmallVec<[_; 8]>>();

        unsafe {
            self.device
                .handle()
                .queue_submit(
                    self.handle,
                    &[vk::SubmitInfoBuilder::new()
                        .wait_semaphores(&wait_semaphores)
                        .wait_dst_stage_mask(&wait_stages)
                        .signal_semaphores(&signal_semaphores)
                        .command_buffers(&[command_buffer.handle()])],
                    fence.map(|fence| fence.handle()),
                )
                .unwrap()
        }
    }

    pub fn present(&mut self, swapchain_image: SwapchainImage) {
        unsafe {
            self.device
                .handle()
                .queue_present_khr(
                    self.handle,
                    &PresentInfoKHRBuilder::new()
                        .swapchains(&[swapchain_image.handle()])
                        .wait_semaphores(&[swapchain_image.info().signal.handle()])
                        .image_indices(&[swapchain_image.index()]),
                )
                .unwrap();
        }
    }

    pub fn cleanup(&mut self, device: &Device) {
        unsafe { device.handle().destroy_command_pool(Some(self.pool), None) }
    }
}
