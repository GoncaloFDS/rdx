use crate::render_pass::RENDERPASS_SMALLVEC_ATTACHMENTS;
use crate::resources::{ImageView, RenderPass};
use erupt::vk;
use smallvec::SmallVec;

pub struct FramebufferInfo {
    pub render_pass: RenderPass,
    pub views: SmallVec<[ImageView; RENDERPASS_SMALLVEC_ATTACHMENTS]>,
    pub extent: vk::Extent2D,
}
