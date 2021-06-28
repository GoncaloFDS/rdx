use crate::resources::DescriptorSetLayout;
use erupt::vk;

pub use self::graphics_pipeline::*;
pub use self::ray_tracing::*;

mod graphics_pipeline;
mod ray_tracing;

pub struct PipelineLayoutInfo {
    pub sets: Vec<DescriptorSetLayout>,
    pub push_constants: Vec<PushConstant>,
}

pub struct PushConstant {
    pub stages: vk::ShaderStageFlags,
    pub offset: u32,
    pub size: u32,
}
