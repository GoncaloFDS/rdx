use erupt::{vk, DeviceLoader};
use std::sync::Arc;

pub struct CommandPool {
    device: Arc<DeviceLoader>,
    pool: vk::CommandPool,
    queue: vk::Queue,
}

impl CommandPool {
    pub fn new(
        device: Arc<DeviceLoader>,
        queue: vk::Queue,
        queue_index: u32,
        create_flags: vk::CommandPoolCreateFlags,
    ) -> Self {
        let pool = unsafe {
            device
                .create_command_pool(
                    &vk::CommandPoolCreateInfoBuilder::new()
                        .queue_family_index(queue_index)
                        .flags(create_flags),
                    None,
                    None,
                )
                .unwrap()
        };
        CommandPool {
            device,
            pool,
            queue,
        }
    }

    pub fn create_command_buffers(
        &self,
        level: vk::CommandBufferLevel,
        count: u32,
    ) -> Vec<vk::CommandBuffer> {
        let cmd_buf_allocate_info = vk::CommandBufferAllocateInfoBuilder::new()
            .command_pool(self.pool)
            .level(level)
            .command_buffer_count(count);
        unsafe {
            self.device
                .allocate_command_buffers(&cmd_buf_allocate_info)
                .unwrap()
        }
    }

    pub fn submit(&self, command_buffers: &[vk::CommandBuffer]) {
        for command_buffer in command_buffers {
            unsafe { self.device.end_command_buffer(*command_buffer).unwrap() }
        }

        let submit_info = vk::SubmitInfoBuilder::new().command_buffers(command_buffers);

        unsafe {
            self.device
                .queue_submit(self.queue, &[submit_info], None)
                .unwrap()
        }
    }

    pub fn submit_and_wait(&self, command_buffers: &[vk::CommandBuffer]) {
        self.submit(command_buffers);
        unsafe {
            self.device.queue_wait_idle(self.queue).unwrap();
            self.device.free_command_buffers(self.pool, command_buffers);
        }
    }

    pub fn destroy(&self) {
        unsafe {
            self.device.destroy_command_pool(Some(self.pool), None);
        }
    }
}
