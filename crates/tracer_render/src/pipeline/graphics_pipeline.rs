use crate::resources::{PipelineLayout, RenderPass};
use crate::shader::Shader;
use erupt::vk;

pub struct GraphicsPipelineInfo {
    pub vertex_bindings: Vec<VertexInputBinding>,
    pub vertex_attributes: Vec<VertexInputAttribute>,
    pub primitive_topology: Vec<vk::PrimitiveTopology>,
    pub vertex_shader: Shader,
    pub rasterizer: Option<Rasterizer>,
    pub layout: PipelineLayout,
    pub render_pass: RenderPass,
    pub subpass: u32,
}

pub struct VertexInputBinding {
    pub rate: VertexInputRate,
    pub stride: u32,
}

pub enum VertexInputRate {
    Vertex,
    Instance,
}

pub struct VertexInputAttribute {
    pub location: u32,
    pub format: vk::Format,
    pub binding: u32,
    pub offset: u32,
}

pub struct Rasterizer {
    pub viewport: vk::Viewport,
    pub scissor: vk::Rect2D,
    pub depth_clamp: bool,
    pub front_face: vk::FrontFace,
    pub culling: vk::CullModeFlags,
    pub polygon_mode: vk::PolygonMode,

    // pub depth_test: Option<DepthTest>,
    // pub stencil_tests: Option<StencilTests>,
    // pub depth_bounds: Option<Bounds>,
    // pub color_blend: ColorBlend,
    pub fragment_shader: Option<Shader>,
}
