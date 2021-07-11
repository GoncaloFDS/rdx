use crate::descriptor::{
    DescriptorSetInfo, DescriptorSetLayoutBinding, DescriptorSetLayoutInfo, DescriptorType,
    Descriptors, WriteDescriptorSet,
};
use crate::framebuffer::FramebufferInfo;
use crate::image::{Image, ImageInfo, ImageView, ImageViewInfo};
use crate::pipeline::{GraphicsPipelineInfo, PipelineLayoutInfo, Rasterizer};
use crate::render_context::RenderContext;
use crate::render_pass::{AttachmentInfo, ClearValue, RenderPassInfo, Subpass};
use crate::renderer::Pass;
use crate::resources::{
    DescriptorSet, Fence, Framebuffer, GraphicsPipeline, PipelineLayout, RenderPass, Sampler,
    Semaphore,
};
use crate::shader::{Shader, ShaderModuleInfo};
use bevy::ecs::schedule::ShouldRun::No;
use bumpalo::Bump;
use erupt::vk;
use erupt::vk::{PipelineStageFlags, ShaderStageFlags};
use lru::LruCache;
use smallvec::smallvec;

pub struct Input {
    pub initial_image: Image,
    pub final_image: Image,
}

pub struct Output;

pub struct TonemapPass {
    render_pass: RenderPass,
    pipeline_layout: PipelineLayout,
    graphics_pipeline: GraphicsPipeline,

    descriptor_sets: [DescriptorSet; 2],
    initial_images: [Option<ImageView>; 2],
    sampler: Sampler,

    framebuffers: LruCache<Image, Framebuffer>,

    vertex_shader: Shader,
    fragment_shader: Shader,
}
impl Pass<'_> for TonemapPass {
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
    ) -> Output {
        let framebuffer = match self.framebuffers.get(&input.final_image) {
            None => {
                let final_image_view = render_context.create_image_view(ImageViewInfo::new(
                    input.final_image.clone(),
                    vk::ImageAspectFlags::COLOR,
                ));

                let framebuffer = render_context.create_framebuffer(FramebufferInfo {
                    render_pass: self.render_pass.clone(),
                    views: smallvec![final_image_view],
                    extent: input.final_image.info().extent,
                });

                self.framebuffers
                    .put(input.final_image, framebuffer.clone());
                framebuffer
            }
            Some(framebuffer) => framebuffer.clone(),
        };

        let mut write_descriptor_sets = vec![];

        let frame_id = (frame % 2) as usize;
        let descriptor_set = &self.descriptor_sets[frame_id];

        match &self.initial_images[frame_id] {
            None => {
                self.initial_images[frame_id] = None;
                let initial_image = render_context.create_image_view(ImageViewInfo::new(
                    input.initial_image.clone(),
                    vk::ImageAspectFlags::COLOR,
                ));
                let initial_image = self.initial_images[frame_id].get_or_insert(initial_image);
                write_descriptor_sets.push(WriteDescriptorSet {
                    descriptor_set,
                    binding: 0,
                    element: 0,
                    descriptors: Descriptors::CombinedImageSampler(bump.alloc([(
                        initial_image.clone(),
                        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                        self.sampler.clone(),
                    )])),
                })
            }
            Some(_) => {}
        }

        render_context.update_descriptor_sets(&write_descriptor_sets, &[]);

        let mut encoder = render_context.queue.create_enconder();

        encoder.begin_render_pass(
            &self.render_pass,
            &framebuffer,
            &[ClearValue::Color(0.5, 0.2, 0.2, 0.0)],
        );

        encoder.bind_graphics_pipeline(&self.graphics_pipeline);
        encoder.bind_descriptor_sets(
            vk::PipelineBindPoint::GRAPHICS,
            &self.pipeline_layout,
            0,
            std::slice::from_ref(descriptor_set),
            &[],
        );

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

impl TonemapPass {
    pub fn new(
        render_context: &RenderContext,
        surface_format: vk::Format,
        extent: vk::Extent2D,
    ) -> Self {
        let descriptor_set_layout =
            render_context.create_descriptor_set_layout(DescriptorSetLayoutInfo {
                bindings: vec![
                    // Initial Image
                    DescriptorSetLayoutBinding {
                        binding: 0,
                        descriptor_type: DescriptorType::CombinedImageSampler,
                        count: 1,
                        stages: ShaderStageFlags::FRAGMENT,
                        flags: vk::DescriptorBindingFlags::empty(),
                    },
                ],
                flags: vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL,
            });

        let vertex_shader = Shader::new(
            render_context.create_shader_module(ShaderModuleInfo::new("tonemap.vert.spv")),
            vk::ShaderStageFlagBits::VERTEX,
        );

        let fragment_shader = Shader::new(
            render_context.create_shader_module(ShaderModuleInfo::new("tonemap.frag.spv")),
            vk::ShaderStageFlagBits::FRAGMENT,
        );

        let render_pass = render_context.create_render_pass(RenderPassInfo {
            attachments: smallvec![AttachmentInfo {
                format: surface_format,
                samples: vk::SampleCountFlags::_1,
                load_op: vk::AttachmentLoadOp::CLEAR,
                store_op: vk::AttachmentStoreOp::STORE,
                initial_layout: None,
                final_layout: vk::ImageLayout::PRESENT_SRC_KHR,
            }],
            subpasses: smallvec![Subpass {
                colors: smallvec![0],
                depth: None,
            }],
        });

        let pipeline_layout = render_context.create_pipeline_layout(PipelineLayoutInfo {
            sets: vec![descriptor_set_layout.clone()],
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

        let descriptor_sets = [
            render_context.create_descriptor_set(DescriptorSetInfo {
                layout: descriptor_set_layout.clone(),
            }),
            render_context.create_descriptor_set(DescriptorSetInfo {
                layout: descriptor_set_layout.clone(),
            }),
        ];

        let sampler = render_context.create_sampler();

        TonemapPass {
            render_pass,
            pipeline_layout,
            graphics_pipeline,
            descriptor_sets,
            initial_images: [None, None],
            sampler,
            framebuffers: LruCache::new(4),
            vertex_shader,
            fragment_shader,
        }
    }
}
