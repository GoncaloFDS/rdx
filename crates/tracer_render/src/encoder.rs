use crate::acceleration_structures::AccelerationStructureBuildGeometryInfo;
use crate::command_buffer::CommandBuffer;
use crate::pipeline::ShaderBindingTable;
use crate::resources::{Buffer, Framebuffer, RayTracingPipeline, RenderPass};
use crevice::internal::bytemuck::Pod;
use erupt::vk;
use winit::event::VirtualKeyCode::Comma;

pub struct Encoder<'a> {
    inner: EncoderInner<'a>,
    command_buffer: CommandBuffer,
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

    pub fn update_buffer<T>(&mut self, buffer: &'a Buffer, offset: u64, data: &'a [T])
    where
        T: Pod,
    {
        todo!()
    }

    pub fn build_acceleration_structure(
        &mut self,
        infos: &'a [AccelerationStructureBuildGeometryInfo<'a>],
    ) {
        todo!()
    }
}

pub struct EncoderInner<'a> {
    commands: Vec<Command<'a>>,
}

impl<'a> EncoderInner<'a> {
    pub fn bind_ray_tracing_pipeline(&mut self, pipeline: &'a RayTracingPipeline) {
        self.commands
            .push(Command::BindRayTracingPipeline { pipeline })
    }

    pub fn trace_rays(&mut self, shader_binding_table: &'a ShaderBindingTable) {
        self.commands.push(Command::TraceRays {
            shader_binding_table,
        })
    }
}

pub enum Command<'a> {
    BindRayTracingPipeline {
        pipeline: &'a RayTracingPipeline,
    },
    TraceRays {
        shader_binding_table: &'a ShaderBindingTable,
    },
}
