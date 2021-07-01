use crate::resources::{PipelineLayout, RenderPass};
use crate::shader::Shader;
use erupt::vk;

#[derive(Clone)]
pub struct GraphicsPipelineInfo {
    pub vertex_bindings: Vec<VertexInputBinding>,
    pub vertex_attributes: Vec<VertexInputAttribute>,
    pub primitive_topology: vk::PrimitiveTopology,
    pub vertex_shader: Shader,
    pub rasterizer: Option<Rasterizer>,
    pub layout: PipelineLayout,
    pub render_pass: RenderPass,
    pub subpass: u32,
}

#[derive(Clone)]
pub struct VertexInputBinding {
    pub input_rate: vk::VertexInputRate,
    pub stride: u32,
}

#[derive(Clone)]
pub struct VertexInputAttribute {
    pub location: u32,
    pub format: vk::Format,
    pub binding: u32,
    pub offset: u32,
}

#[derive(Clone)]
pub struct Rasterizer {
    pub viewport: vk::Viewport,
    pub depth_clamp: bool,
    pub front_face: vk::FrontFace,
    pub cull_mode: vk::CullModeFlags,
    pub polygon_mode: vk::PolygonMode,

    pub fragment_shader: Option<Shader>,
}
