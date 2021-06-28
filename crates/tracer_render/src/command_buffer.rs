use crate::encoder::Command;
use crate::queue::QueueId;
use erupt::vk;

pub struct CommandBuffer {
    handle: vk::CommandBuffer,
    queue: QueueId,
    recording: bool,
}

impl CommandBuffer {
    pub fn new(handle: vk::CommandBuffer, queue: QueueId) -> Self {
        CommandBuffer {
            handle,
            queue,
            recording: false,
        }
    }

    pub fn write(&mut self, commands: &[Command<'_>]) {
        todo!()
    }
}
