use crate::buffer::BufferInfo;
use crate::descriptor::{DescriptorSetInfo, DescriptorSetLayoutInfo, DescriptorSizes};
use crate::pipeline::{GraphicsPipelineInfo, PipelineLayoutInfo};
use crate::render_pass::RenderPassInfo;
use crate::resources::{
    Buffer, DescriptorSet, DescriptorSetLayout, Fence, GraphicsPipeline, MappableBuffer,
    PipelineLayout, RenderPass, Semaphore, ShaderModule,
};
use crate::shader::{ShaderLanguage, ShaderModuleInfo};
use crate::surface::Surface;
use crate::swapchain::Swapchain;
use crevice::internal::bytemuck::Pod;
use erupt::vk1_0::ImageLayout;
use erupt::{vk, DeviceLoader, InstanceLoader};
use gpu_alloc::{GpuAllocator, UsageFlags};
use gpu_alloc_erupt::EruptMemoryDevice;
use parking_lot::Mutex;
use slab::Slab;
use smallvec::SmallVec;
use std::ffi::{CStr, CString};
use std::sync::Arc;

pub struct DeviceInner {
    handle: DeviceLoader,
    instance: Arc<InstanceLoader>,
    physical_device: vk::PhysicalDevice,
    allocator: Mutex<GpuAllocator<vk::DeviceMemory>>,
    buffers: Mutex<Slab<vk::Buffer>>,
    swapchains: Mutex<Slab<vk::SwapchainKHR>>,
    semaphores: Mutex<Slab<vk::Semaphore>>,
    fences: Mutex<Slab<vk::Fence>>,
    framebuffers: Mutex<Slab<vk::Framebuffer>>,
    images: Mutex<Slab<vk::Image>>,
    image_views: Mutex<Slab<vk::ImageView>>,
    samplers: Mutex<Slab<vk::Sampler>>,
    descriptor_pools: Mutex<Slab<vk::DescriptorPool>>,
    descriptor_set_layouts: Mutex<Slab<vk::DescriptorSetLayout>>,
    pipelines: Mutex<Slab<vk::Pipeline>>,
    pipeline_layouts: Mutex<Slab<vk::PipelineLayout>>,
    render_passes: Mutex<Slab<vk::RenderPass>>,
    shaders: Mutex<Slab<vk::ShaderModule>>,
    acceleration_structures: Mutex<Slab<vk::AccelerationStructureKHR>>,
}

#[derive(Clone)]
pub struct Device {
    inner: Arc<DeviceInner>,
}

impl Device {
    pub fn new(
        instance: Arc<InstanceLoader>,
        device: DeviceLoader,
        physical_device: vk::PhysicalDevice,
    ) -> Self {
        let allocator = Mutex::new(GpuAllocator::new(
            gpu_alloc::Config::i_am_prototyping(),
            unsafe { gpu_alloc_erupt::device_properties(&instance, physical_device).unwrap() },
        ));
        Device {
            inner: Arc::new(DeviceInner {
                handle: device,
                instance,
                physical_device,
                allocator,
                buffers: Mutex::new(Slab::with_capacity(1024)),
                swapchains: Mutex::new(Slab::with_capacity(1024)),
                semaphores: Mutex::new(Slab::with_capacity(1024)),
                fences: Mutex::new(Slab::with_capacity(1024)),
                framebuffers: Mutex::new(Slab::with_capacity(1024)),
                images: Mutex::new(Slab::with_capacity(1024)),
                image_views: Mutex::new(Slab::with_capacity(1024)),
                samplers: Mutex::new(Slab::with_capacity(1024)),
                descriptor_pools: Mutex::new(Slab::with_capacity(1024)),
                descriptor_set_layouts: Mutex::new(Slab::with_capacity(1024)),
                pipelines: Mutex::new(Slab::with_capacity(1024)),
                pipeline_layouts: Mutex::new(Slab::with_capacity(1024)),
                render_passes: Mutex::new(Slab::with_capacity(1024)),
                shaders: Mutex::new(Slab::with_capacity(1024)),
                acceleration_structures: Mutex::new(Slab::with_capacity(1024)),
            }),
        }
    }

    pub fn cleanup(&mut self) {
        let device = self.handle();

        unsafe {
            self.inner
                .swapchains
                .lock()
                .iter()
                .for_each(|(_, &swapchain)| device.destroy_swapchain_khr(Some(swapchain), None));

            self.inner
                .semaphores
                .lock()
                .iter()
                .for_each(|(_, &semaphore)| device.destroy_semaphore(Some(semaphore), None));

            self.inner
                .fences
                .lock()
                .iter()
                .for_each(|(_, &fence)| device.destroy_fence(Some(fence), None));

            self.handle().destroy_device(None)
        }
    }

    pub fn instance(&self) -> &InstanceLoader {
        &self.inner.instance
    }

    pub fn handle(&self) -> &DeviceLoader {
        &self.inner.handle
    }

    pub fn swapchains(&self) -> &Mutex<Slab<vk::SwapchainKHR>> {
        &self.inner.swapchains
    }

    pub fn create_buffer(&self, info: BufferInfo, allocation_flags: UsageFlags) -> MappableBuffer {
        let buffer = unsafe {
            self.inner
                .handle
                .create_buffer(
                    &vk::BufferCreateInfoBuilder::new()
                        .size(info.size)
                        .usage(info.usage_flags)
                        .sharing_mode(vk::SharingMode::EXCLUSIVE),
                    None,
                )
                .unwrap()
        };

        let mem_requirements = unsafe { self.inner.handle.get_buffer_memory_requirements(buffer) };

        let mem_block = unsafe {
            self.inner
                .allocator
                .lock()
                .alloc(
                    EruptMemoryDevice::wrap(&self.inner.handle),
                    gpu_alloc::Request {
                        size: mem_requirements.size,
                        align_mask: (mem_requirements.alignment - 1) | info.align,
                        usage: allocation_flags,
                        memory_types: mem_requirements.memory_type_bits,
                    },
                )
                .unwrap()
        };

        unsafe {
            self.inner
                .handle
                .bind_buffer_memory(buffer, *mem_block.memory(), mem_block.offset())
                .unwrap()
        }

        let device_address = if allocation_flags.contains(UsageFlags::DEVICE_ADDRESS) {
            let device_address = unsafe {
                self.inner.handle.get_buffer_device_address(
                    &vk::BufferDeviceAddressInfoBuilder::new().buffer(buffer),
                )
            };
            Some(device_address)
        } else {
            None
        };

        let buffer_index = self.inner.buffers.lock().insert(buffer);

        tracing::debug!("Created Buffer {:p}", buffer);
        MappableBuffer::new(
            info,
            buffer,
            device_address,
            buffer_index,
            mem_block,
            allocation_flags,
        )
    }

    pub fn create_buffer_with_data<T: 'static>(&self, info: BufferInfo, data: &[T]) -> Buffer
    where
        T: Pod,
    {
        let mut buffer = self.create_buffer(info, UsageFlags::UPLOAD);

        unsafe {
            let ptr = buffer
                .memory_block()
                .map(
                    EruptMemoryDevice::wrap(&self.inner.handle),
                    0,
                    std::mem::size_of_val(data),
                )
                .expect("Mapping to buffer failed");

            std::ptr::copy_nonoverlapping(
                data.as_ptr() as *const u8,
                ptr.as_ptr(),
                std::mem::size_of_val(data),
            );

            buffer
                .memory_block()
                .unmap(EruptMemoryDevice::wrap(&self.inner.handle));
        }
        buffer.into()
    }

    pub fn create_swapchain(&self, surface: &Surface) -> Swapchain {
        Swapchain::new(self, surface)
    }

    pub fn create_semaphore(&self) -> Semaphore {
        let semaphore = unsafe {
            self.handle()
                .create_semaphore(&vk::SemaphoreCreateInfoBuilder::new(), None)
                .unwrap()
        };

        self.inner.semaphores.lock().insert(semaphore);

        Semaphore::new(semaphore)
    }

    pub fn create_fence(&self) -> Fence {
        let fence = unsafe {
            self.handle()
                .create_fence(&vk::FenceCreateInfoBuilder::new(), None)
                .unwrap()
        };
        self.inner.fences.lock().insert(fence);

        Fence::new(fence)
    }

    pub fn create_descriptor_set_layout(
        &self,
        info: DescriptorSetLayoutInfo,
    ) -> DescriptorSetLayout {
        let handle = unsafe {
            self.handle()
                .create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfoBuilder::new()
                        .bindings(
                            &info
                                .bindings
                                .iter()
                                .map(|binding| {
                                    vk::DescriptorSetLayoutBindingBuilder::new()
                                        .binding(binding.binding)
                                        .descriptor_count(binding.count)
                                        .descriptor_type(binding.descriptor_type)
                                        .stage_flags(binding.stages)
                                })
                                .collect::<SmallVec<[_; 16]>>(),
                        )
                        .flags(info.flags),
                    None,
                )
                .unwrap()
        };

        self.inner.descriptor_set_layouts.lock().insert(handle);

        let sizes = DescriptorSizes::from_bindings(&info.bindings);

        DescriptorSetLayout::new(info, handle, sizes)
    }

    pub fn create_descriptor_set(&self, info: DescriptorSetInfo) -> DescriptorSet {
        let pool_flags = if info
            .layout
            .info()
            .flags
            .contains(vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL)
        {
            vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND
        } else {
            vk::DescriptorPoolCreateFlags::empty()
        };

        let pool = unsafe {
            self.handle()
                .create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfoBuilder::new()
                        .max_sets(1)
                        .pool_sizes(&info.layout.sizes())
                        .flags(pool_flags),
                    None,
                )
                .unwrap()
        };

        let handles = unsafe {
            self.handle()
                .allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfoBuilder::new()
                        .descriptor_pool(pool)
                        .set_layouts(&[info.layout.handle()]),
                )
                .unwrap()
        };

        self.inner.descriptor_pools.lock().insert(pool);

        DescriptorSet::new(info, handles[0], pool)
    }

    pub fn create_pipeline_layout(&self, info: PipelineLayoutInfo) -> PipelineLayout {
        let pipeline_layout = unsafe {
            self.handle()
                .create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfoBuilder::new()
                        .set_layouts(
                            &info
                                .sets
                                .iter()
                                .map(|set| set.handle())
                                .collect::<SmallVec<[_; 16]>>(),
                        )
                        .push_constant_ranges(
                            &info
                                .push_constants
                                .iter()
                                .map(|push_constants| {
                                    vk::PushConstantRangeBuilder::new()
                                        .stage_flags(push_constants.stages)
                                        .offset(push_constants.offset)
                                        .size(push_constants.size)
                                })
                                .collect::<SmallVec<[_; 16]>>(),
                        ),
                    None,
                )
                .unwrap()
        };

        self.inner.pipeline_layouts.lock().insert(pipeline_layout);

        PipelineLayout::new(info, pipeline_layout)
    }

    pub fn create_shader_module(&self, info: ShaderModuleInfo) -> ShaderModule {
        let code = match info.language {
            ShaderLanguage::GLSL => panic!("glsl is not supported"),
            ShaderLanguage::SPIRV => &*info.code,
        };

        let spv = erupt::utils::decode_spv(&code).unwrap();
        let module = unsafe {
            self.handle()
                .create_shader_module(&vk::ShaderModuleCreateInfoBuilder::new().code(&spv), None)
                .unwrap()
        };

        self.inner.shaders.lock().insert(module);

        ShaderModule::new(info, module)
    }

    pub fn create_render_pass(&self, info: RenderPassInfo) -> RenderPass {
        let attachments = info
            .attachments
            .iter()
            .map(|attachment| {
                vk::AttachmentDescriptionBuilder::new()
                    .format(attachment.format)
                    .samples(vk::SampleCountFlagBits::_1)
                    .load_op(attachment.load_op)
                    .store_op(attachment.store_op)
                    .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                    .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                    .initial_layout(match attachment.initial_layout {
                        None => vk::ImageLayout::UNDEFINED,
                        Some(layout) => layout,
                    })
                    .final_layout(attachment.final_layout)
            })
            .collect::<SmallVec<[_; 16]>>();

        let mut subpass_attachments = Vec::new();
        let subpass_offsets = {
            info.subpasses
                .iter()
                .enumerate()
                .map(|(i, subpass)| {
                    let color_offset = subpass_attachments.len();
                    subpass_attachments.extend(
                        subpass
                            .colors
                            .iter()
                            .enumerate()
                            .map(|(color_i, &color)| {
                                vk::AttachmentReferenceBuilder::new()
                                    .attachment(color as _)
                                    .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                            })
                            .collect::<SmallVec<[_; 16]>>(),
                    );

                    let depth_offset = subpass_attachments.len();
                    if let Some(depth) = subpass.depth {
                        subpass_attachments.push(
                            vk::AttachmentReferenceBuilder::new()
                                .attachment(depth as _)
                                .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL),
                        )
                    }
                    (color_offset, depth_offset)
                })
                .collect::<SmallVec<[_; 16]>>()
        };

        let subpasses = info
            .subpasses
            .iter()
            .zip(subpass_offsets)
            .map(|(subpass, (color_offset, depth_offset))| {
                let subpass_descriptor = vk::SubpassDescriptionBuilder::new()
                    .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
                    .color_attachments(&subpass_attachments[color_offset..depth_offset]);

                if subpass.depth.is_some() {
                    subpass_descriptor.depth_stencil_attachment(&subpass_attachments[depth_offset])
                } else {
                    subpass_descriptor
                }
            })
            .collect::<Vec<_>>();

        let render_pass_create_info = vk::RenderPassCreateInfoBuilder::new()
            .attachments(&attachments)
            .subpasses(&subpasses);

        let render_pass = unsafe {
            self.handle()
                .create_render_pass(&render_pass_create_info, None)
                .unwrap()
        };

        self.inner.render_passes.lock().insert(render_pass);

        RenderPass::new(info, render_pass)
    }

    pub fn create_graphics_pipeline(&self, info: GraphicsPipelineInfo) -> GraphicsPipeline {
        let mut shader_stages = Vec::with_capacity(2);

        let vertex_binding_descriptions = info
            .vertex_bindings
            .iter()
            .enumerate()
            .map(|(i, binding)| {
                vk::VertexInputBindingDescriptionBuilder::new()
                    .binding(i as _)
                    .stride(binding.stride)
                    .input_rate(binding.input_rate)
            })
            .collect::<SmallVec<[_; 16]>>();

        let vertex_attribute_descriptions = info
            .vertex_attributes
            .iter()
            .map(|attribute| {
                vk::VertexInputAttributeDescriptionBuilder::new()
                    .location(attribute.location)
                    .binding(attribute.binding)
                    .offset(attribute.offset)
                    .format(attribute.format)
            })
            .collect::<SmallVec<[_; 16]>>();

        let vertex_input_state = vk::PipelineVertexInputStateCreateInfoBuilder::new()
            .vertex_binding_descriptions(&vertex_binding_descriptions)
            .vertex_attribute_descriptions(&vertex_attribute_descriptions);

        let shader_entry_point = CString::new("main").unwrap();

        shader_stages.push(
            vk::PipelineShaderStageCreateInfoBuilder::new()
                .stage(vk::ShaderStageFlagBits::VERTEX)
                .module(info.vertex_shader.module.handle())
                .name(&shader_entry_point),
        );

        let input_assembly_state = vk::PipelineInputAssemblyStateCreateInfoBuilder::new()
            .topology(info.primitive_topology)
            .primitive_restart_enable(false);

        let dynamic_state_info;
        let viewport_info;
        let rasterization_info;
        let depth_stencil_info;
        let color_blend_attachments;
        let color_blend_info;
        let multisample_info;

        let pipeline_info = if let Some(rasterizer) = &info.rasterizer {
            dynamic_state_info = vk::PipelineDynamicStateCreateInfoBuilder::new()
                .dynamic_states(&[vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR]);
            viewport_info = vk::PipelineViewportStateCreateInfoBuilder::new()
                .viewport_count(1)
                .scissor_count(1);
            rasterization_info = vk::PipelineRasterizationStateCreateInfoBuilder::new()
                .rasterizer_discard_enable(false)
                .depth_clamp_enable(rasterizer.depth_clamp)
                .polygon_mode(rasterizer.polygon_mode)
                .cull_mode(rasterizer.cull_mode)
                .front_face(rasterizer.front_face)
                .line_width(1.0);
            let stencil_op = vk::StencilOpStateBuilder::new()
                .fail_op(vk::StencilOp::KEEP)
                .pass_op(vk::StencilOp::KEEP)
                .compare_op(vk::CompareOp::ALWAYS)
                .build();
            depth_stencil_info = vk::PipelineDepthStencilStateCreateInfoBuilder::new()
                .depth_test_enable(true)
                .depth_write_enable(true)
                .depth_compare_op(vk::CompareOp::LESS_OR_EQUAL)
                .depth_bounds_test_enable(false)
                .stencil_test_enable(false)
                .front(stencil_op)
                .back(stencil_op);
            color_blend_attachments = [vk::PipelineColorBlendAttachmentStateBuilder::new()
                .color_write_mask(
                    vk::ColorComponentFlags::R
                        | vk::ColorComponentFlags::G
                        | vk::ColorComponentFlags::B
                        | vk::ColorComponentFlags::A,
                )];
            color_blend_info = vk::PipelineColorBlendStateCreateInfoBuilder::new()
                .attachments(&color_blend_attachments);
            multisample_info = vk::PipelineMultisampleStateCreateInfoBuilder::new()
                .rasterization_samples(vk::SampleCountFlagBits::_1);

            vk::GraphicsPipelineCreateInfoBuilder::new()
                .stages(&shader_stages)
                .vertex_input_state(&vertex_input_state)
                .input_assembly_state(&input_assembly_state)
                .layout(info.layout.handle())
                .render_pass(info.render_pass.handle())
                .subpass(info.subpass)
                .rasterization_state(&rasterization_info)
                .dynamic_state(&dynamic_state_info)
                .viewport_state(&viewport_info)
                .multisample_state(&multisample_info)
                .color_blend_state(&color_blend_info)
                .depth_stencil_state(&depth_stencil_info)
        } else {
            vk::GraphicsPipelineCreateInfoBuilder::new()
                .stages(&shader_stages)
                .vertex_input_state(&vertex_input_state)
                .input_assembly_state(&input_assembly_state)
                .layout(info.layout.handle())
                .render_pass(info.render_pass.handle())
                .subpass(info.subpass)
        };

        let pipelines = unsafe {
            self.handle()
                .create_graphics_pipelines(None, &[pipeline_info], None)
                .unwrap()
        };

        let pipeline = pipelines[0];
        self.inner.pipelines.lock().insert(pipeline);

        GraphicsPipeline::new(info, pipeline)
    }
}
