mod raster;

use crate::render_context::RenderContext;
use crate::resources::Semaphore;
use erupt::vk;

pub use raster::*;

pub trait Pass<'a> {
    type Input;
    type Output;

    fn draw(
        &mut self,
        input: Self::Input,
        frame: u64,
        wait: &[(vk::PipelineStageFlags, Semaphore)],
        signal: &[Semaphore],
        render_context: &mut RenderContext,
    ) -> Self::Output;
}
