use crate::image::ImageView;
use crate::render_pass::DEFAULT_ATTACHMENT_COUNT;
use crate::resources::RenderPass;
use erupt::vk;
use smallvec::SmallVec;

#[derive(Clone)]
pub struct FramebufferInfo {
    pub render_pass: RenderPass,
    pub views: SmallVec<[ImageView; DEFAULT_ATTACHMENT_COUNT]>,
    pub extent: vk::Extent2D,
}
