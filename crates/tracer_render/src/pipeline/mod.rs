pub use self::graphics_pipeline::*;
pub use self::raster_pipeline::*;
pub use self::ray_tracing::*;

use crate::image::Image;
use crate::render_context::RenderContext;
use crate::resources::{DescriptorSetLayout, Semaphore};
use erupt::vk;

mod graphics_pipeline;
mod raster_pipeline;
mod ray_tracing;

pub trait Pipeline {
    fn draw(
        &mut self,
        target: Image,
        target_wait: &Semaphore,
        target_signal: &Semaphore,
        render_context: &mut RenderContext,
    );
}

#[derive(Clone)]
pub struct PipelineLayoutInfo {
    pub sets: Vec<DescriptorSetLayout>,
    pub push_constants: Vec<PushConstant>,
}

#[derive(Clone)]
pub struct PushConstant {
    pub stages: vk::ShaderStageFlags,
    pub offset: u32,
    pub size: u32,
}
