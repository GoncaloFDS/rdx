pub mod raster_pass;

use crate::render_context::RenderContext;
use crate::resources::{Fence, Semaphore};
use erupt::vk;

pub use raster_pass::*;

pub trait Pass<'a> {
    type Input;
    type Output;

    fn draw(
        &mut self,
        input: Self::Input,
        frame: u64,
        wait: &[(vk::PipelineStageFlags, Semaphore)],
        signal: &[Semaphore],
        fence: Option<&Fence>,
        render_context: &mut RenderContext,
    ) -> Self::Output;
}
