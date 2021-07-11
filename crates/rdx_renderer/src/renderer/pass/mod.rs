pub mod raster_pass;
pub mod raytracing_pass;
pub mod tonemap_pass;

use crate::render_context::RenderContext;
use crate::resources::{Fence, Semaphore};
use erupt::vk;

use bumpalo::Bump;
use erupt::vk1_0::PipelineStageFlags;
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
        bump: &Bump,
    ) -> Self::Output;
}
