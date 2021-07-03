use erupt::vk;
use smallvec::SmallVec;

pub const DEFAULT_ATTACHMENT_COUNT: usize = 4;
pub const DEFAULT_SUBPASS_COUNT: usize = 1;

#[derive(Clone)]
pub struct RenderPassInfo {
    pub attachments: SmallVec<[AttachmentInfo; DEFAULT_ATTACHMENT_COUNT]>,
    pub subpasses: SmallVec<[Subpass; DEFAULT_SUBPASS_COUNT]>,
}

#[derive(Clone)]
pub struct AttachmentInfo {
    pub format: vk::Format,
    pub samples: vk::SampleCountFlags,
    pub load_op: vk::AttachmentLoadOp,
    pub store_op: vk::AttachmentStoreOp,
    pub initial_layout: Option<vk::ImageLayout>,
    pub final_layout: vk::ImageLayout,
}

#[derive(Clone)]
pub struct Subpass {
    pub colors: SmallVec<[usize; DEFAULT_ATTACHMENT_COUNT]>,
    pub depth: Option<usize>,
}

#[derive(Clone)]
pub struct SubpassDependency {
    pub src: Option<usize>,
    pub dst: Option<usize>,
    pub src_stages: vk::PipelineStageFlags,
    pub dst_stages: vk::PipelineStageFlags,
}

pub enum ClearValue {
    Color(f32, f32, f32, f32),
    DepthStencil(f32, u32),
}
