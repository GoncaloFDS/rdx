use crate::buffer::BufferRegion;
use crate::shader::Shader;
use erupt::vk::PipelineLayout;

#[derive(Clone)]
pub struct RayTracingPipelineInfo {
    pub shaders: Vec<Shader>,
    pub groups: Vec<RayTracingShaderGroupInfo>,
    pub max_recursion_depth: u32,
    pub layout: PipelineLayout,
}

#[derive(Clone)]
pub enum RayTracingShaderGroupInfo {
    Raygen {
        raygen: u32,
    },
    Miss {
        miss: u32,
    },
    Triangle {
        any_hit: Option<u32>,
        closest_hit: Option<u32>,
    },
}

#[derive(Clone)]
pub struct ShaderBindingTableInfo<'a> {
    pub raygen: Option<u32>,
    pub miss: &'a [u32],
    pub hit: &'a [u32],
    pub callable: &'a [u32],
}

pub struct ShaderBindingTable {
    pub raygen: Option<BufferRegion>,
    pub miss: Option<BufferRegion>,
    pub hit: Option<BufferRegion>,
    pub callable: Option<BufferRegion>,
}
