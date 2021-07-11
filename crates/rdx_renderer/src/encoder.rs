use crate::acceleration_structures::AccelerationStructureBuildGeometryInfo;
use crate::command_buffer::CommandBuffer;
use crate::device::Device;
use crate::image::ImageMemoryBarrier;
use crate::pipeline::ShaderBindingTable;
use crate::render_pass::ClearValue;
use crate::resources::{
    Buffer, DescriptorSet, Framebuffer, GraphicsPipeline, PipelineLayout, RayTracingPipeline,
    RenderPass,
};
use crevice::internal::bytemuck::Pod;
use erupt::vk;
use erupt::vk1_0::PipelineBindPoint;
use std::ops::{Deref, DerefMut, Range};

pub struct Encoder<'a> {
    inner: EncoderInner<'a>,
    command_buffer: CommandBuffer,
}

impl<'a> Deref for Encoder<'a> {
    type Target = EncoderInner<'a>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<'a> DerefMut for Encoder<'a> {
    fn deref_mut(&mut self) -> &mut EncoderInner<'a> {
        &mut self.inner
    }
}

impl<'a> Encoder<'a> {
    pub fn new(command_buffer: CommandBuffer) -> Self {
        Encoder {
            inner: EncoderInner {
                commands: Vec::new(),
            },
            command_buffer,
        }
    }

    pub fn finish(mut self, device: &Device) -> CommandBuffer {
        self.command_buffer.write(device, &self.inner.commands);

        self.command_buffer
    }
}

pub struct EncoderInner<'a> {
    commands: Vec<Command<'a>>,
}

impl<'a> EncoderInner<'a> {
    pub fn begin_render_pass(
        &mut self,
        pass: &'a RenderPass,
        framebuffer: &'a Framebuffer,
        clears: &'a [ClearValue],
    ) {
        self.commands.push(Command::BeginRenderPass {
            render_pass: pass,
            framebuffer,
            clears,
        })
    }

    pub fn end_render_pass(&mut self) {
        self.commands.push(Command::EndRenderPass)
    }

    pub fn bind_graphics_pipeline(&mut self, pipeline: &'a GraphicsPipeline) {
        self.commands
            .push(Command::BindGraphicsPipeline { pipeline })
    }

    pub fn bind_ray_tracing_pipeline(&mut self, pipeline: &'a RayTracingPipeline) {
        self.commands
            .push(Command::BindRayTracingPipeline { pipeline })
    }

    pub fn bind_descriptor_sets(
        &mut self,
        bind_point: vk::PipelineBindPoint,
        layout: &'a PipelineLayout,
        first_set: u32,
        descriptor_sets: &'a [DescriptorSet],
        dynamic_offsets: &'a [u32],
    ) {
        self.commands.push(Command::BindDescriptorSets {
            bind_point,
            layout,
            first_set,
            descriptor_sets,
            dynamic_offsets,
        });
    }

    pub fn set_viewport(&mut self, viewport: vk::Viewport) {
        self.commands.push(Command::SetViewport { viewport })
    }

    pub fn set_scissor(&mut self, scissor: vk::Rect2D) {
        self.commands.push(Command::SetScissor { scissor })
    }

    pub fn draw(&mut self, vertices: Range<u32>, instances: Range<u32>) {
        self.commands.push(Command::Draw {
            vertices,
            instances,
        })
    }

    pub fn draw_indexed(&mut self, indices: Range<u32>, vertex_offset: i32, instances: Range<u32>) {
        self.commands.push(Command::DrawIndexed {
            indices,
            vertex_offset,
            instances,
        });
    }

    pub fn update_buffer<T>(&mut self, buffer: &'a Buffer, offset: u64, data: &'a [T])
    where
        T: Pod,
    {
        let data = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data))
        };

        self.commands.push(Command::UpdateBuffer {
            buffer,
            offset,
            data,
        })
    }

    pub fn bind_vertex_buffers(&mut self, first: u32, buffers: &'a [(Buffer, u64)]) {
        self.commands
            .push(Command::BindVertexBuffers { first, buffers })
    }

    pub fn bind_index_buffer(
        &mut self,
        buffer: &'a Buffer,
        offset: u64,
        index_type: vk::IndexType,
    ) {
        self.commands.push(Command::BindIndexBuffer {
            buffer,
            offset,
            index_type,
        })
    }

    pub fn build_acceleration_structure(
        &mut self,
        infos: &'a [AccelerationStructureBuildGeometryInfo<'a>],
    ) {
        if infos.is_empty() {
            return;
        }

        self.commands
            .push(Command::BuildAccelerationStructure { infos })
    }

    pub fn trace_rays(
        &mut self,
        shader_binding_table: &'a ShaderBindingTable,
        extent: vk::Extent2D,
    ) {
        self.commands.push(Command::TraceRays {
            shader_binding_table,
            extent,
        })
    }
    pub fn pipeline_barrier(
        &mut self,
        src: vk::PipelineStageFlags,
        dst: vk::PipelineStageFlags,
        src_access_mask: vk::AccessFlags,
        dst_access_mask: vk::AccessFlags,
        image_barriers: &'a [ImageMemoryBarrier<'a>],
    ) {
        self.commands.push(Command::PipelineBarrier {
            src,
            dst,
            src_access_mask,
            dst_access_mask,
            image_barriers,
        });
    }
}

pub enum Command<'a> {
    BeginRenderPass {
        render_pass: &'a RenderPass,
        framebuffer: &'a Framebuffer,
        clears: &'a [ClearValue],
    },
    EndRenderPass,

    BindGraphicsPipeline {
        pipeline: &'a GraphicsPipeline,
    },

    BindRayTracingPipeline {
        pipeline: &'a RayTracingPipeline,
    },

    BindDescriptorSets {
        bind_point: vk::PipelineBindPoint,
        layout: &'a PipelineLayout,
        first_set: u32,
        descriptor_sets: &'a [DescriptorSet],
        dynamic_offsets: &'a [u32],
    },

    SetViewport {
        viewport: vk::Viewport,
    },

    SetScissor {
        scissor: vk::Rect2D,
    },

    Draw {
        vertices: Range<u32>,
        instances: Range<u32>,
    },

    DrawIndexed {
        indices: Range<u32>,
        vertex_offset: i32,
        instances: Range<u32>,
    },

    UpdateBuffer {
        buffer: &'a Buffer,
        offset: u64,
        data: &'a [u8],
    },

    BindVertexBuffers {
        first: u32,
        buffers: &'a [(Buffer, u64)],
    },

    BindIndexBuffer {
        buffer: &'a Buffer,
        offset: u64,
        index_type: vk::IndexType,
    },

    BuildAccelerationStructure {
        infos: &'a [AccelerationStructureBuildGeometryInfo<'a>],
    },

    TraceRays {
        shader_binding_table: &'a ShaderBindingTable,
        extent: vk::Extent2D,
    },

    PipelineBarrier {
        src: vk::PipelineStageFlags,
        dst: vk::PipelineStageFlags,
        src_access_mask: vk::AccessFlags,
        dst_access_mask: vk::AccessFlags,
        image_barriers: &'a [ImageMemoryBarrier<'a>],
    },
}
