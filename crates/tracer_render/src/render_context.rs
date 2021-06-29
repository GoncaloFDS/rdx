use crate::device::Device;
use crate::queue::Queue;
use std::ops::Deref;

pub struct RenderContext {
    pub device: Device,
    pub queue: Queue,
}

impl Deref for RenderContext {
    type Target = Device;

    fn deref(&self) -> &Self::Target {
        &self.device
    }
}

impl RenderContext {
    pub fn new(device: Device, queue: Queue) -> Self {
        RenderContext { device, queue }
    }
}
