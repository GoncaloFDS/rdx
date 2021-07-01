use crate::pipeline::Pipeline;
use crate::render_context::RenderContext;
use crate::renderer::{Pass, RasterPass};
use crate::resources::{Fence, Image, Semaphore};

pub struct RasterPipeline {
    raster_pass: RasterPass,
    frame: u64,
    fences: [Fence; 2],
}

impl RasterPipeline {
    pub fn new(render_context: &RenderContext) -> Self {
        RasterPipeline {
            raster_pass: RasterPass::new(render_context),
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
        self.raster_pass
            .draw((), self.frame, &[], &[], render_context)
    }
}
