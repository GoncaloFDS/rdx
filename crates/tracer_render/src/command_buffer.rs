use crate::device::Device;
use crate::encoder::Command;
use erupt::vk;

pub struct CommandBuffer {
    handle: vk::CommandBuffer,
    queue_family_index: u32,
    recording: bool,
}

impl CommandBuffer {
    pub fn new(handle: vk::CommandBuffer, queue_family_index: u32) -> Self {
        CommandBuffer {
            handle,
            queue_family_index,
            recording: false,
        }
    }

    pub fn handle(&self) -> vk::CommandBuffer {
        self.handle
    }

    pub fn write(&mut self, device: &Device, commands: &[Command<'_>]) {
        let device = device.handle();
        if !self.recording {
            unsafe {
                device
                    .begin_command_buffer(
                        self.handle,
                        &vk::CommandBufferBeginInfoBuilder::new()
                            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                    )
                    .unwrap()
            }
            self.recording = true;
        }

        for command in commands {
            match *command {
                Command::BeginRenderPass {
                    render_pass,
                    framebuffer,
                    clears,
                } => unsafe {
                    let clear_values = [vk::ClearValue {
                        color: vk::ClearColorValue {
                            float32: [0.5, 0.25, 0.25, 1.0],
                        },
                    }];
                    device.cmd_begin_render_pass(
                        self.handle,
                        &vk::RenderPassBeginInfoBuilder::new()
                            .render_pass(render_pass.handle())
                            .framebuffer(framebuffer.handle())
                            .render_area(vk::Rect2D {
                                offset: vk::Offset2D { x: 0, y: 0 },
                                extent: framebuffer.info().extent,
                            })
                            .clear_values(&clear_values),
                        vk::SubpassContents::INLINE,
                    )
                },
                Command::EndRenderPass => unsafe { device.cmd_end_render_pass(self.handle) },
                Command::BindGraphicsPipeline { .. } => unimplemented!(),
                Command::BindRayTracingPipeline { .. } => unimplemented!(),
                Command::BindGraphicsDescriptorSets { .. } => unimplemented!(),
                Command::BindRayTracingDescriptorSets { .. } => unimplemented!(),
                Command::Draw { .. } => unimplemented!(),
                Command::DrawIndexed { .. } => unimplemented!(),
                Command::UpdateBuffer { .. } => unimplemented!(),
                Command::BindVertexBuffers { .. } => unimplemented!(),
                Command::BindIndexBuffer { .. } => unimplemented!(),
                Command::BuildAccelerationStructure { .. } => unimplemented!(),
                Command::TraceRays { .. } => unimplemented!(),
            }
        }

        unsafe {
            device.end_command_buffer(self.handle).unwrap();
        }
    }
}
