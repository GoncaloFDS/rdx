use erupt::vk;
use smallvec::SmallVec;

pub const RENDERPASS_SMALLVEC_ATTACHMENTS: usize = 8;
pub const SMALLVEC_SUBPASSES: usize = 4;

pub struct RenderPassInfo {
    pub attachments: SmallVec<[AttachmentInfo; RENDERPASS_SMALLVEC_ATTACHMENTS]>,
    pub subpasses: SmallVec<[Subpass; SMALLVEC_SUBPASSES]>,
    pub dependencies: SmallVec<[SubpassDependency; SMALLVEC_SUBPASSES]>,
}

pub struct AttachmentInfo {
    pub format: vk::Format,
    pub samples: vk::SampleCountFlags,
    pub load_op: vk::AttachmentLoadOp,
    pub store_op: vk::AttachmentStoreOp,
    pub initial_layout: Option<vk::ImageLayout>,
    pub final_layout: vk::ImageLayout,
}

pub struct Subpass {
    pub colors: SmallVec<[usize; RENDERPASS_SMALLVEC_ATTACHMENTS]>,
    pub depth: Option<usize>,
}

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
