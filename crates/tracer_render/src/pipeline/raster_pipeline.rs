use crate::pipeline::Pipeline;
use crate::render_context::RenderContext;
use crate::renderer::{raster_pass, Pass, RasterPass};
use crate::resources::{Fence, Image, Semaphore};
use erupt::vk;
use erupt::vk1_0::{Extent2D, Format};

pub struct RasterPipeline {
    raster_pass: RasterPass,
    frame: u64,
    fences: [Fence; 2],
}

impl RasterPipeline {
    pub fn new(
        render_context: &RenderContext,
        surface_format: vk::Format,
        extent: vk::Extent2D,
    ) -> Self {
        RasterPipeline {
            raster_pass: RasterPass::new(render_context, surface_format, extent),
            frame: 0,
            fences: [render_context.create_fence(), render_context.create_fence()],
        }
    }
}

impl Pipeline for RasterPipeline {
    fn draw(
        &mut self,
        target: Image,
        target_wait: &Semaphore,
        target_signal: &Semaphore,
        render_context: &mut RenderContext,
    ) {
        let fence = &self.fences[(self.frame % 2) as usize];
        if self.frame > 1 {
            render_context.wait_fences(&[fence], true);
            render_context.reset_fences(&[fence]);
        }
        self.raster_pass.draw(
            raster_pass::Input {
                target: target.clone(),
            },
            self.frame,
            &[(
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                target_wait.clone(),
            )],
            &[target_signal.clone()],
            Some(fence),
            render_context,
        );

        self.frame += 1;
    }
}
