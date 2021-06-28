use crate::device::Device;
use crate::encoder::Encoder;
use erupt::vk;

pub struct Queue {
    handle: vk::Queue,
    pool: vk::CommandPool,
    device: Device,
}

impl Queue {
    pub fn new(handle: vk::Queue, pool: vk::CommandPool, device: Device) -> Self {
        Queue {
            handle,
            pool,
            device,
        }
    }

    pub fn create_enconder(&mut self) -> Encoder<'static> {
        todo!()
    }

    pub fn submit(&self) {
        todo!()
    }
}

pub struct QueueId {
    pub family: usize,
    pub index: usize,
}
