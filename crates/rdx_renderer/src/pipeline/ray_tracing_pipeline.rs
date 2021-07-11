use crate::buffer::BufferRegion;
use crate::image::Image;
use crate::pipeline::Pipeline;
use crate::render_context::RenderContext;
use crate::renderer::raytracing_pass::RayTracingPass;
use crate::renderer::tonemap_pass::TonemapPass;
use crate::renderer::{raster_pass, raytracing_pass, tonemap_pass, Pass};
use crate::resources::{AccelerationStructure, Fence, PipelineLayout, Semaphore};
use crate::shader::Shader;
use bumpalo::Bump;
use erupt::vk;
use std::collections::HashMap;

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

pub struct PathTracingPipeline {
    raytracing_pass: RayTracingPass,
    tonemap_pass: TonemapPass,
    frame: u64,
    fences: [Fence; 2],
}

impl PathTracingPipeline {
    pub fn new(
        render_context: &RenderContext,
        surface_format: vk::Format,
        extent: vk::Extent2D,
    ) -> Self {
        PathTracingPipeline {
            raytracing_pass: RayTracingPass::new(render_context, extent),
            tonemap_pass: TonemapPass::new(render_context, surface_format, extent),
            frame: 0,
            fences: [render_context.create_fence(), render_context.create_fence()],
        }
    }
}

impl Pipeline for PathTracingPipeline {
    fn draw(
        &mut self,
        render_context: &mut RenderContext,
        target: Image,
        target_wait: &Semaphore,
        target_signal: &Semaphore,
        blases: &HashMap<u8, AccelerationStructure>,
        bump: &Bump,
    ) {
        let fence = &self.fences[(self.frame % 2) as usize];
        if self.frame > 1 {
            render_context.wait_fences(&[fence], true);
            render_context.reset_fences(&[fence]);
        }

        let raytracing_output = self.raytracing_pass.draw(
            raytracing_pass::Input { blases },
            self.frame,
            &[],
            &[],
            None,
            render_context,
            bump,
        );

        self.tonemap_pass.draw(
            tonemap_pass::Input {
                initial_image: raytracing_output.output_image.clone(),
                final_image: target.clone(),
            },
            self.frame,
            &[(
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                target_wait.clone(),
            )],
            &[target_signal.clone()],
            Some(fence),
            render_context,
            bump,
        );

        self.frame += 1;
    }
}
