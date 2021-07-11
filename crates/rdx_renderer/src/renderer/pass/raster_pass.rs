use crate::descriptor::{DescriptorSetInfo, DescriptorSetLayoutBinding, DescriptorSetLayoutInfo};
use crate::framebuffer::FramebufferInfo;
use crate::image::{Image, ImageInfo, ImageViewInfo};
use crate::pipeline::{GraphicsPipelineInfo, PipelineLayoutInfo, Rasterizer};
use crate::render_context::RenderContext;
use crate::render_pass::{AttachmentInfo, ClearValue, RenderPassInfo, Subpass};
use crate::renderer::Pass;
use crate::resources::{
    Fence, Framebuffer, GraphicsPipeline, PipelineLayout, RenderPass, Semaphore,
};
use crate::shader::{Shader, ShaderModuleInfo};
use bumpalo::Bump;
use erupt::vk;
use erupt::vk1_0::{Extent2D, Format, PipelineStageFlags};
use lru::LruCache;
use smallvec::smallvec;

pub struct RasterPass {
    render_pass: RenderPass,
    pipeline_layout: PipelineLayout,
    graphics_pipeline: GraphicsPipeline,

    framebuffers: LruCache<Image, Framebuffer>,

    depth_image: Image,

    vertex_shader: Shader,
    fragment_shader: Shader,
}

pub struct Input {
    pub target: Image,
}

pub struct Output;

impl Pass<'_> for RasterPass {
    type Input = Input;
    type Output = Output;

    fn draw(
        &mut self,
        input: Input,
        frame: u64,
        wait: &[(PipelineStageFlags, Semaphore)],
        signal: &[Semaphore],
        fence: Option<&Fence>,
        render_context: &mut RenderContext,
        bump: &Bump,
    ) -> Self::Output {
        let fb;
        let framebuffer = match self.framebuffers.get(&input.target) {
            None => {
                let color_view = render_context.create_image_view(ImageViewInfo::new(
                    input.target.clone(),
                    vk::ImageAspectFlags::COLOR,
                ));

                let depth_view = render_context.create_image_view(ImageViewInfo::new(
                    self.depth_image.clone(),
                    vk::ImageAspectFlags::DEPTH,
                ));

                fb = render_context.create_framebuffer(FramebufferInfo {
                    render_pass: self.render_pass.clone(),
                    views: smallvec![color_view, depth_view],
                    extent: input.target.info().extent,
                });

                self.framebuffers.put(input.target, fb.clone());
                &fb
            }
            Some(framebuffer) => framebuffer,
        };

        let mut encoder = render_context.queue.create_enconder();

        encoder.begin_render_pass(
            &self.render_pass,
            framebuffer,
            &[
                ClearValue::Color(0.5, 0.2, 0.2, 0.0),
                ClearValue::DepthStencil(1.0, 0),
            ],
        );

        encoder.bind_graphics_pipeline(&self.graphics_pipeline);

        encoder.set_viewport(vk::Viewport {
            x: 0.0,
            y: framebuffer.info().extent.height as f32,
            width: framebuffer.info().extent.width as f32,
            height: -(framebuffer.info().extent.height as f32),
            min_depth: 0.0,
            max_depth: 1.0,
        });

        encoder.set_scissor(vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: framebuffer.info().extent,
        });

        encoder.draw(0..3, 0..1);

        encoder.end_render_pass();

        let command_buffer = encoder.finish(&render_context.device);

        render_context
            .queue
            .submit(command_buffer, wait, signal, fence);

        Output
    }
}

impl RasterPass {
    pub fn new(
        render_context: &RenderContext,
        surface_format: vk::Format,
        extent: vk::Extent2D,
    ) -> Self {
        let vertex_shader = Shader::new(
            render_context.create_shader_module(ShaderModuleInfo::new("shader.vert.spv")),
            vk::ShaderStageFlagBits::VERTEX,
        );

        let fragment_shader = Shader::new(
            render_context.create_shader_module(ShaderModuleInfo::new("shader.frag.spv")),
            vk::ShaderStageFlagBits::FRAGMENT,
        );

        let depth_image = render_context.create_image(ImageInfo {
            extent,
            format: vk::Format::D32_SFLOAT,
            mip_levels: 1,
            array_layers: 1,
            samples: vk::SampleCountFlagBits::_1,
            usage: vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
        });

        let render_pass = render_context.create_render_pass(RenderPassInfo {
            attachments: smallvec![
                AttachmentInfo {
                    format: surface_format,
                    samples: vk::SampleCountFlags::_1,
                    load_op: vk::AttachmentLoadOp::CLEAR,
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
                    width: extent.width as _,
                    height: extent.height as _,
                    min_depth: 0.0,
                    max_depth: 1.0,
                },
                depth_clamp: false,
                front_face: vk::FrontFace::COUNTER_CLOCKWISE,
                cull_mode: vk::CullModeFlags::NONE,
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
            depth_image,
            vertex_shader,
            fragment_shader,
        }
    }
}
