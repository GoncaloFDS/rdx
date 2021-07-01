use crate::descriptor::{DescriptorSetInfo, DescriptorSetLayoutBinding, DescriptorSetLayoutInfo};
use crate::pipeline::{GraphicsPipelineInfo, PipelineLayoutInfo, Rasterizer};
use crate::render_context::RenderContext;
use crate::render_pass::{AttachmentInfo, RenderPassInfo, Subpass};
use crate::renderer::Pass;
use crate::resources::{
    Framebuffer, GraphicsPipeline, Image, PipelineLayout, RenderPass, Semaphore,
};
use crate::shader::{Shader, ShaderLanguage, ShaderModuleInfo};
use erupt::vk;
use erupt::vk1_0::PipelineStageFlags;
use lru::LruCache;
use smallvec::smallvec;

pub struct Input;
pub struct Output;

pub struct RasterPass {
    render_pass: RenderPass,
    pipeline_layout: PipelineLayout,
    graphics_pipeline: GraphicsPipeline,

    framebuffers: LruCache<Image, Framebuffer>,

    vertex_shader: Shader,
    fragment_shader: Shader,
}

impl Pass<'_> for RasterPass {
    type Input = ();
    type Output = ();

    fn draw(
        &mut self,
        input: Self::Input,
        frame: u64,
        wait: &[(PipelineStageFlags, Semaphore)],
        signal: &[Semaphore],
        render_context: &mut RenderContext,
    ) -> Self::Output {
        todo!()
    }
}

impl RasterPass {
    pub fn new(render_context: &RenderContext) -> Self {
        let vertex_shader = {
            let module = render_context.create_shader_module(ShaderModuleInfo::new(
                "shader.vert.spv",
                ShaderLanguage::SPIRV,
            ));
            Shader::new(module, vk::ShaderStageFlags::VERTEX)
        };

        let fragment_shader = {
            let module = render_context.create_shader_module(ShaderModuleInfo::new(
                "shader.frag.spv",
                ShaderLanguage::SPIRV,
            ));
            Shader::new(module, vk::ShaderStageFlags::FRAGMENT)
        };

        let render_pass = render_context.create_render_pass(RenderPassInfo {
            attachments: smallvec![
                AttachmentInfo {
                    format: vk::Format::B8G8R8A8_UNORM,
                    samples: vk::SampleCountFlags::_1,
                    load_op: vk::AttachmentLoadOp::DONT_CARE,
                    store_op: vk::AttachmentStoreOp::STORE,
                    initial_layout: None,
                    final_layout: vk::ImageLayout::PRESENT_SRC_KHR
                },
                AttachmentInfo {
                    format: vk::Format::D32_SFLOAT,
                    samples: vk::SampleCountFlags::_1,
                    load_op: vk::AttachmentLoadOp::CLEAR,
                    store_op: vk::AttachmentStoreOp::DONT_CARE,
                    initial_layout: None,
                    final_layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL
                },
            ],
            subpasses: smallvec![Subpass {
                colors: smallvec![0],
                depth: Some(1),
            }],
        });

        let pipeline_layout = render_context.create_pipeline_layout(PipelineLayoutInfo {
            sets: vec![],
            push_constants: vec![],
        });

        let graphics_pipeline = render_context.create_graphics_pipeline(GraphicsPipelineInfo {
            vertex_bindings: vec![],
            vertex_attributes: vec![],
            primitive_topology: vk::PrimitiveTopology::TRIANGLE_LIST,
            vertex_shader: vertex_shader.clone(),
            rasterizer: Some(Rasterizer {
                viewport: vk::Viewport {
                    x: 0.0,
                    y: 0.0,
                    width: 800.0,
                    height: 600.0,
                    min_depth: 0.0,
                    max_depth: 1000.0,
                },
                depth_clamp: false,
                front_face: vk::FrontFace::COUNTER_CLOCKWISE,
                cull_mode: vk::CullModeFlags::BACK,
                polygon_mode: vk::PolygonMode::FILL,
                fragment_shader: Some(fragment_shader.clone()),
            }),
            layout: pipeline_layout.clone(),
            render_pass: render_pass.clone(),
            subpass: 0,
        });

        RasterPass {
            render_pass,
            pipeline_layout,
            graphics_pipeline,
            framebuffers: LruCache::new(4),
            vertex_shader,
            fragment_shader,
        }
    }
}
