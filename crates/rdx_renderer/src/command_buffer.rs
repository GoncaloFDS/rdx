use crate::device::Device;
use crate::encoder::Command;
use crate::render_pass::{ClearValue, DEFAULT_ATTACHMENT_COUNT};
use erupt::vk;
use smallvec::SmallVec;

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
                    let mut clears = clears.into_iter();
                    let clear_values = render_pass
                        .info()
                        .attachments
                        .iter()
                        .map(|attachment| {
                            let clear = clears.next().unwrap();
                            match *clear {
                                ClearValue::Color(r, g, b, a) => vk::ClearValue {
                                    color: vk::ClearColorValue {
                                        float32: [r, g, b, a],
                                    },
                                },
                                ClearValue::DepthStencil(depth, stencil) => vk::ClearValue {
                                    depth_stencil: vk::ClearDepthStencilValue { depth, stencil },
                                },
                            }
                        })
                        .collect::<SmallVec<[_; DEFAULT_ATTACHMENT_COUNT]>>();

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
                Command::BindGraphicsPipeline { pipeline } => unsafe {
                    device.cmd_bind_pipeline(
                        self.handle,
                        vk::PipelineBindPoint::GRAPHICS,
                        pipeline.handle(),
                    )
                },
                Command::BindRayTracingPipeline { .. } => unimplemented!(),
                Command::BindGraphicsDescriptorSets { .. } => unimplemented!(),
                Command::BindRayTracingDescriptorSets { .. } => unimplemented!(),
                Command::Draw {
                    ref vertices,
                    ref instances,
                } => unsafe {
                    device.cmd_draw(
                        self.handle,
                        vertices.end - vertices.start,
                        instances.end - instances.start,
                        vertices.start,
                        instances.start,
                    )
                },
                Command::SetViewport { viewport } => unsafe {
                    device.cmd_set_viewport(self.handle, 0, &[viewport.into_builder()])
                },
                Command::SetScissor { scissor } => unsafe {
                    device.cmd_set_scissor(self.handle, 0, &[scissor.into_builder()])
                },
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
