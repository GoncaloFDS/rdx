use crate::acceleration_structures::{
    AccelerationStructureBuildSizesInfo, AccelerationStructureGeometryInfo,
    AccelerationStructureInfo, AccelerationStructureLevel,
};
use crate::buffer::{BufferInfo, BufferRegion, DeviceAddress};
use crate::descriptor::{
    CopyDescriptorSet, DescriptorSetInfo, DescriptorSetLayoutInfo, DescriptorSizes, Descriptors,
    WriteDescriptorSet,
};
use crate::framebuffer::FramebufferInfo;
use crate::image::{Image, ImageInfo, ImageView, ImageViewInfo};
use crate::physical_device::PhysicalDevice;
use crate::pipeline::{
    GraphicsPipelineInfo, PipelineLayoutInfo, RayTracingPipelineInfo, RayTracingShaderGroupInfo,
    ShaderBindingTable, ShaderBindingTableInfo,
};
use crate::render_pass::RenderPassInfo;
use crate::resources::{
    AccelerationStructure, Buffer, DescriptorSet, DescriptorSetLayout, Fence, Framebuffer,
    GraphicsPipeline, PipelineLayout, RayTracingPipeline, RenderPass, Sampler, Semaphore,
    ShaderModule,
};
use crate::shader::ShaderModuleInfo;
use crate::surface::Surface;
use crate::swapchain::Swapchain;
use crate::util::{align_up, ToErupt};
use crevice::internal::bytemuck;
use crevice::internal::bytemuck::Pod;
use erupt::{vk, DeviceLoader, ExtendableFromConst, InstanceLoader};
use gpu_alloc::{GpuAllocator, UsageFlags};
use gpu_alloc_erupt::EruptMemoryDevice;
use parking_lot::Mutex;
use slab::Slab;
use smallvec::SmallVec;
use std::convert::TryFrom;
use std::ffi::CString;
use std::ops::Range;
use std::sync::Arc;

pub struct DeviceInner {
    handle: DeviceLoader,
    instance: Arc<InstanceLoader>,
    physical_device: PhysicalDevice,
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
    shader_modules: Mutex<Slab<vk::ShaderModule>>,
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
        physical_device: PhysicalDevice,
    ) -> Self {
        let allocator = Mutex::new(GpuAllocator::new(
            gpu_alloc::Config::i_am_prototyping(),
            unsafe {
                gpu_alloc_erupt::device_properties(&instance, physical_device.handle()).unwrap()
            },
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
                shader_modules: Mutex::new(Slab::with_capacity(1024)),
                acceleration_structures: Mutex::new(Slab::with_capacity(1024)),
            }),
        }
    }

    pub fn cleanup(&mut self) {
        let device = self.handle();

        unsafe {
            self.inner
                .framebuffers
                .lock()
                .iter()
                .for_each(|(_, &framebuffer)| device.destroy_framebuffer(Some(framebuffer), None));

            self.inner
                .pipeline_layouts
                .lock()
                .iter()
                .for_each(|(_, &pipeline_layout)| {
                    device.destroy_pipeline_layout(Some(pipeline_layout), None)
                });

            self.inner
                .pipelines
                .lock()
                .iter()
                .for_each(|(_, &pipeline)| device.destroy_pipeline(Some(pipeline), None));

            self.inner
                .render_passes
                .lock()
                .iter()
                .for_each(|(_, &render_pass)| device.destroy_render_pass(Some(render_pass), None));

            self.inner
                .shader_modules
                .lock()
                .iter()
                .for_each(|(_, &shader_module)| {
                    device.destroy_shader_module(Some(shader_module), None)
                });

            self.inner
                .image_views
                .lock()
                .iter()
                .for_each(|(_, &view)| device.destroy_image_view(Some(view), None));

            self.inner
                .images
                .lock()
                .iter()
                .for_each(|(_, &image)| device.destroy_image(Some(image), None));

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

    fn allocator(&self) -> &Mutex<GpuAllocator<vk::DeviceMemory>> {
        &self.inner.allocator
    }

    pub fn create_buffer(&self, info: BufferInfo) -> Buffer {
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
                    EruptMemoryDevice::wrap(self.handle()),
                    gpu_alloc::Request {
                        size: mem_requirements.size,
                        align_mask: (mem_requirements.alignment - 1) | info.align,
                        usage: info.allocation_flags,
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

        let device_address = if info.allocation_flags.contains(UsageFlags::DEVICE_ADDRESS) {
            let device_address = unsafe {
                self.inner.handle.get_buffer_device_address(
                    &vk::BufferDeviceAddressInfoBuilder::new().buffer(buffer),
                )
            };
            Some(DeviceAddress::new(device_address))
        } else {
            None
        };

        let buffer_index = self.inner.buffers.lock().insert(buffer);
        let allocation_flags = info.allocation_flags;

        tracing::debug!("Created Buffer {:p}", buffer);
        Buffer::new(
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
        let mut buffer = self.create_buffer(info);

        unsafe {
            let ptr = buffer
                .memory_block()
                .map(
                    EruptMemoryDevice::wrap(self.handle()),
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
                .unmap(EruptMemoryDevice::wrap(self.handle()));
        }
        buffer.into()
    }

    pub fn write_buffer<T>(&self, buffer: &mut Buffer, offset: u64, data: &[T])
    where
        T: Pod,
    {
        unsafe {
            buffer
                .memory_block()
                .write_bytes(
                    EruptMemoryDevice::wrap(self.handle()),
                    offset,
                    bytemuck::cast_slice(data),
                )
                .unwrap();
        }
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

    pub fn reset_fences(&self, fences: &[&Fence]) {
        let fences = fences
            .iter()
            .map(|fence| fence.handle())
            .collect::<SmallVec<[_; 16]>>();
        unsafe {
            self.handle().reset_fences(&fences).unwrap();
        }
    }

    pub fn wait_fences(&self, fences: &[&Fence], wait_all: bool) {
        let fences = fences
            .iter()
            .map(|fence| fence.handle())
            .collect::<SmallVec<[_; 16]>>();
        unsafe {
            self.handle()
                .wait_for_fences(&fences, wait_all, !0)
                .unwrap();
        }
    }

    pub fn wait_idle(&self) {
        unsafe { self.handle().device_wait_idle().unwrap() }
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
                                        .descriptor_type(binding.descriptor_type.to_erupt())
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

    pub fn update_descriptor_sets<'a>(
        &self,
        writes: &[WriteDescriptorSet<'a>],
        copies: &[CopyDescriptorSet<'a>],
    ) {
        debug_assert!(copies.is_empty());

        let mut ranges = SmallVec::<[_; 64]>::new();
        let mut images = SmallVec::<[_; 16]>::new();
        let mut acceleration_structures = SmallVec::<[_; 64]>::new();
        let mut write_descriptor_acceleration_structures = SmallVec::<[_; 16]>::new();

        for write in writes {
            match write.descriptors {
                Descriptors::Sampler(_) => unimplemented!(),
                Descriptors::CombinedImageSampler(slice) => {
                    let start = images.len();
                    images.extend(slice.iter().map(|(image_view, image_layout, sampler)| {
                        vk::DescriptorImageInfoBuilder::new()
                            .sampler(sampler.handle())
                            .image_view(image_view.handle())
                            .image_layout(*image_layout)
                    }));
                    ranges.push(start..images.len());
                }
                Descriptors::SampledImage(_) => unimplemented!(),
                Descriptors::StorageImage(slice) => {
                    let start = images.len();
                    images.extend(slice.iter().map(|(image_view, image_layout)| {
                        vk::DescriptorImageInfoBuilder::new()
                            .image_view(image_view.handle())
                            .image_layout(*image_layout)
                    }));
                    ranges.push(start..images.len());
                }
                Descriptors::UniformBuffer(_) => unimplemented!(),
                Descriptors::StorageBuffer(_) => unimplemented!(),
                Descriptors::UniformBufferDynamic(_) => unimplemented!(),
                Descriptors::StorageBufferDynamic(_) => unimplemented!(),
                Descriptors::InputAttachment(_) => unimplemented!(),
                Descriptors::AccelerationStructure(slice) => {
                    let start = acceleration_structures.len();
                    acceleration_structures.extend(
                        slice
                            .iter()
                            .map(|acceleration_structure| acceleration_structure.handle()),
                    );

                    ranges.push(start..acceleration_structures.len());

                    write_descriptor_acceleration_structures
                        .push(vk::WriteDescriptorSetAccelerationStructureKHRBuilder::new())
                }
            }
        }

        let mut ranges = ranges.into_iter();
        let mut write_descriptor_acceleration_structures =
            write_descriptor_acceleration_structures.iter_mut();

        let writes = writes
            .iter()
            .map(|write| {
                let write_builder = vk::WriteDescriptorSetBuilder::new()
                    .dst_set(write.descriptor_set.handle())
                    .dst_binding(write.binding)
                    .dst_array_element(write.element);

                match write.descriptors {
                    Descriptors::Sampler(_) => unimplemented!(),
                    Descriptors::CombinedImageSampler(_) => write_builder
                        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .image_info(&images[ranges.next().unwrap()]),
                    Descriptors::SampledImage(_) => unimplemented!(),
                    Descriptors::StorageImage(_) => write_builder
                        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                        .image_info(&images[ranges.next().unwrap()]),
                    Descriptors::UniformBuffer(_) => unimplemented!(),
                    Descriptors::StorageBuffer(_) => unimplemented!(),
                    Descriptors::UniformBufferDynamic(_) => unimplemented!(),
                    Descriptors::StorageBufferDynamic(_) => unimplemented!(),
                    Descriptors::InputAttachment(_) => unimplemented!(),
                    Descriptors::AccelerationStructure(_) => {
                        let range = ranges.next().unwrap();
                        let mut write_builder = write_builder
                            .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR);
                        write_builder.descriptor_count = range.len() as u32;

                        let acc_structure_write =
                            write_descriptor_acceleration_structures.next().unwrap();
                        *acc_structure_write =
                            vk::WriteDescriptorSetAccelerationStructureKHRBuilder::new()
                                .acceleration_structures(&acceleration_structures[range.clone()]);
                        write_builder.extend_from(&mut *acc_structure_write)
                    }
                }
            })
            .collect::<SmallVec<[_; 16]>>();

        unsafe { self.handle().update_descriptor_sets(&writes, &[]) }
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
        let spv = erupt::utils::decode_spv(&info.code).unwrap();
        let module = unsafe {
            self.handle()
                .create_shader_module(&vk::ShaderModuleCreateInfoBuilder::new().code(&spv), None)
                .unwrap()
        };

        self.inner.shader_modules.lock().insert(module);

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
                .depth_bias_enable(false)
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

            if let Some(fragment_shader) = &rasterizer.fragment_shader {
                shader_stages.push(
                    vk::PipelineShaderStageCreateInfoBuilder::new()
                        .stage(vk::ShaderStageFlagBits::FRAGMENT)
                        .module(fragment_shader.module.handle())
                        .name(&shader_entry_point),
                )
            }

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

    pub fn create_image(&self, info: ImageInfo) -> Image {
        let image = unsafe {
            self.handle()
                .create_image(
                    &vk::ImageCreateInfoBuilder::new()
                        .image_type(vk::ImageType::_2D)
                        .format(info.format)
                        .extent(vk::Extent3D {
                            width: info.extent.width,
                            height: info.extent.height,
                            depth: 1,
                        })
                        .mip_levels(info.mip_levels)
                        .array_layers(info.array_layers)
                        .samples(info.samples)
                        .tiling(vk::ImageTiling::OPTIMAL)
                        .usage(info.usage)
                        .sharing_mode(vk::SharingMode::EXCLUSIVE)
                        .initial_layout(vk::ImageLayout::UNDEFINED),
                    None,
                )
                .unwrap()
        };

        let memory_requirements = unsafe { self.handle().get_image_memory_requirements(image) };

        let memory_block = unsafe {
            self.allocator()
                .lock()
                .alloc(
                    EruptMemoryDevice::wrap(self.handle()),
                    gpu_alloc::Request {
                        size: memory_requirements.size,
                        align_mask: memory_requirements.alignment - 1,
                        usage: get_allocator_memory_usage(&info.usage),
                        memory_types: memory_requirements.memory_type_bits,
                    },
                )
                .unwrap()
        };

        self.inner.images.lock().insert(image);

        unsafe {
            self.handle()
                .bind_image_memory(image, *memory_block.memory(), memory_block.offset())
                .unwrap();
        }

        Image::new(info, image, Some(memory_block))
    }

    pub fn create_image_view(&self, info: ImageViewInfo) -> ImageView {
        let view = unsafe {
            self.handle()
                .create_image_view(
                    &vk::ImageViewCreateInfoBuilder::new()
                        .image(info.image.handle())
                        .format(info.image.info().format)
                        .view_type(info.view_type)
                        .subresource_range(
                            vk::ImageSubresourceRangeBuilder::new()
                                .aspect_mask(info.subresource.aspect)
                                .base_mip_level(info.subresource.first_level)
                                .level_count(info.subresource.level_count)
                                .base_array_layer(info.subresource.first_layer)
                                .layer_count(info.subresource.layer_count)
                                .build(),
                        ),
                    None,
                )
                .unwrap()
        };

        self.inner.image_views.lock().insert(view);

        ImageView::new(info, view)
    }

    pub fn create_sampler(&self) -> Sampler {
        let sampler = unsafe {
            self.handle()
                .create_sampler(
                    &vk::SamplerCreateInfoBuilder::new()
                        .mag_filter(vk::Filter::NEAREST)
                        .min_filter(vk::Filter::NEAREST)
                        .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
                        .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_BORDER)
                        .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_BORDER)
                        .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_BORDER)
                        .mip_lod_bias(0.0)
                        .anisotropy_enable(false)
                        .compare_enable(false)
                        .compare_op(vk::CompareOp::NEVER)
                        .min_lod(0.0)
                        .max_lod(0.0)
                        .border_color(vk::BorderColor::FLOAT_TRANSPARENT_BLACK)
                        .unnormalized_coordinates(false),
                    None,
                )
                .unwrap()
        };

        self.inner.samplers.lock().insert(sampler);

        Sampler::new(sampler)
    }

    pub fn create_framebuffer(&self, info: FramebufferInfo) -> Framebuffer {
        let render_pass = info.render_pass.handle();

        let attachments = info
            .views
            .iter()
            .map(|view| view.handle())
            .collect::<SmallVec<[_; 16]>>();

        let framebuffer = unsafe {
            self.handle()
                .create_framebuffer(
                    &vk::FramebufferCreateInfoBuilder::new()
                        .render_pass(render_pass)
                        .attachments(&attachments)
                        .width(info.extent.width)
                        .height(info.extent.height)
                        .layers(1),
                    None,
                )
                .unwrap()
        };

        self.inner.framebuffers.lock().insert(framebuffer);

        Framebuffer::new(info, framebuffer)
    }

    pub fn create_ray_tracing_pipeline(&self, info: RayTracingPipelineInfo) -> RayTracingPipeline {
        let shader_entry_name = CString::new("main").unwrap();
        let stages = info
            .shaders
            .iter()
            .map(|shader| {
                vk::PipelineShaderStageCreateInfoBuilder::new()
                    .stage(shader.stage)
                    .module(shader.module.handle())
                    .name(&shader_entry_name)
            })
            .collect::<Vec<_>>();

        let groups = info
            .groups
            .iter()
            .map(|group| {
                let shader_group_info = vk::RayTracingShaderGroupCreateInfoKHRBuilder::new();
                match *group {
                    RayTracingShaderGroupInfo::Raygen { raygen } => shader_group_info
                        ._type(vk::RayTracingShaderGroupTypeKHR::GENERAL_KHR)
                        .general_shader(raygen)
                        .any_hit_shader(vk::SHADER_UNUSED_KHR)
                        .closest_hit_shader(vk::SHADER_UNUSED_KHR)
                        .intersection_shader(vk::SHADER_UNUSED_KHR),
                    RayTracingShaderGroupInfo::Miss { miss } => shader_group_info
                        ._type(vk::RayTracingShaderGroupTypeKHR::GENERAL_KHR)
                        .general_shader(miss)
                        .any_hit_shader(vk::SHADER_UNUSED_KHR)
                        .closest_hit_shader(vk::SHADER_UNUSED_KHR)
                        .intersection_shader(vk::SHADER_UNUSED_KHR),
                    RayTracingShaderGroupInfo::Triangle {
                        any_hit,
                        closest_hit,
                    } => shader_group_info
                        ._type(vk::RayTracingShaderGroupTypeKHR::TRIANGLES_HIT_GROUP_KHR)
                        .general_shader(vk::SHADER_UNUSED_KHR)
                        .any_hit_shader(any_hit.unwrap_or(vk::SHADER_UNUSED_KHR))
                        .closest_hit_shader(closest_hit.unwrap_or(vk::SHADER_UNUSED_KHR))
                        .intersection_shader(vk::SHADER_UNUSED_KHR),
                }
            })
            .collect::<Vec<_>>();

        let pipeline = unsafe {
            self.handle()
                .create_ray_tracing_pipelines_khr(
                    None,
                    None,
                    &[vk::RayTracingPipelineCreateInfoKHRBuilder::new()
                        .stages(&stages)
                        .groups(&groups)
                        .max_pipeline_ray_recursion_depth(info.max_recursion_depth)
                        .layout(info.layout.handle())],
                    None,
                )
                .unwrap()[0]
        };

        let group_size = self
            .inner
            .physical_device
            .info()
            .raytracing_properties
            .shader_group_handle_size as usize;

        let total_size = group_size.checked_mul(info.groups.len()).unwrap();

        let group_count = info.groups.len() as u32;

        let mut bytes = vec![0u8; total_size];

        unsafe {
            self.handle()
                .get_ray_tracing_shader_group_handles_khr(
                    pipeline,
                    0,
                    group_count,
                    bytes.len(),
                    bytes.as_mut_ptr() as *mut _,
                )
                .unwrap();
        }

        self.inner.pipelines.lock().insert(pipeline);

        RayTracingPipeline::new(info, pipeline, bytes.into())
    }

    pub fn create_shader_binding_table(
        &self,
        pipeline: &RayTracingPipeline,
        info: ShaderBindingTableInfo,
    ) -> ShaderBindingTable {
        let rt_properties = self.inner.physical_device.info().raytracing_properties;

        let group_size = rt_properties.shader_group_handle_size as u64;
        let group_align = (rt_properties.shader_group_base_alignment - 1) as u64;

        let group_count = (info.raygen.is_some() as usize
            + info.miss.len()
            + info.hit.len()
            + info.callable.len()) as u32;

        let group_stride = align_up(group_align, group_size).unwrap() as u64;

        let total_size = group_stride.checked_mul(group_count as _).unwrap() as usize;

        let mut bytes = vec![0; total_size];

        let mut write_offset = 0;

        let group_handlers = pipeline.group_handlers();

        let raygen_handlers = copy_group_handlers(
            group_handlers,
            &mut bytes,
            info.raygen.iter().copied(),
            &mut write_offset,
            group_size,
            group_stride as usize,
        );

        let miss_handlers = copy_group_handlers(
            group_handlers,
            &mut bytes,
            info.miss.iter().copied(),
            &mut write_offset,
            group_size,
            group_stride as usize,
        );

        let hit_handlers = copy_group_handlers(
            group_handlers,
            &mut bytes,
            info.hit.iter().copied(),
            &mut write_offset,
            group_size,
            group_stride as usize,
        );

        let callable_handlers = copy_group_handlers(
            group_handlers,
            &mut bytes,
            info.callable.iter().copied(),
            &mut write_offset,
            group_size,
            group_stride as usize,
        );

        let sbt_buffer = self.create_buffer_with_data(
            BufferInfo {
                align: group_align,
                size: total_size as _,
                usage_flags: vk::BufferUsageFlags::SHADER_BINDING_TABLE_KHR
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                allocation_flags: gpu_alloc::UsageFlags::DEVICE_ADDRESS
                    | gpu_alloc::UsageFlags::HOST_ACCESS,
            },
            &bytes,
        );

        ShaderBindingTable {
            raygen: raygen_handlers.map(|range| BufferRegion {
                buffer: sbt_buffer.clone(),
                offset: range.start,
                size: range.end - range.start,
                stride: Some(group_stride),
            }),
            miss: miss_handlers.map(|range| BufferRegion {
                buffer: sbt_buffer.clone(),
                offset: range.start,
                size: range.end - range.start,
                stride: Some(group_stride),
            }),
            hit: hit_handlers.map(|range| BufferRegion {
                buffer: sbt_buffer.clone(),
                offset: range.start,
                size: range.end - range.start,
                stride: Some(group_stride),
            }),
            callable: callable_handlers.map(|range| BufferRegion {
                buffer: sbt_buffer.clone(),
                offset: range.start,
                size: range.end - range.start,
                stride: Some(group_stride),
            }),
        }
    }

    pub fn get_acceleration_structure_build_sizes(
        &self,
        level: AccelerationStructureLevel,
        flags: vk::BuildAccelerationStructureFlagsKHR,
        geometry: &[AccelerationStructureGeometryInfo],
    ) -> AccelerationStructureBuildSizesInfo {
        let geometries = geometry
            .iter()
            .map(|info| match *info {
                AccelerationStructureGeometryInfo::Triangles {
                    max_vertex_count,
                    vertex_format,
                    index_type,
                    ..
                } => vk::AccelerationStructureGeometryKHRBuilder::new()
                    .geometry_type(vk::GeometryTypeKHR::TRIANGLES_KHR)
                    .geometry(vk::AccelerationStructureGeometryDataKHR {
                        triangles: vk::AccelerationStructureGeometryTrianglesDataKHRBuilder::new()
                            .max_vertex(max_vertex_count)
                            .vertex_format(vertex_format)
                            .index_type(index_type)
                            .build(),
                    }),
                AccelerationStructureGeometryInfo::Instances { .. } => {
                    vk::AccelerationStructureGeometryKHRBuilder::new()
                        .geometry_type(vk::GeometryTypeKHR::INSTANCES_KHR)
                        .geometry(vk::AccelerationStructureGeometryDataKHR {
                            instances: vk::AccelerationStructureGeometryInstancesDataKHR::default(),
                        })
                }
            })
            .collect::<SmallVec<[_; 4]>>();

        let max_primitive_counts = geometry
            .iter()
            .map(|info| match *info {
                AccelerationStructureGeometryInfo::Triangles {
                    max_primitive_count,
                    ..
                } => max_primitive_count,
                AccelerationStructureGeometryInfo::Instances {
                    max_primitive_count,
                    ..
                } => max_primitive_count,
            })
            .collect::<SmallVec<[_; 4]>>();

        let build_info = vk::AccelerationStructureBuildGeometryInfoKHRBuilder::new()
            ._type(level.to_erupt())
            .flags(flags)
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD_KHR)
            .geometries(&geometries);

        let build_sizes = unsafe {
            self.handle().get_acceleration_structure_build_sizes_khr(
                vk::AccelerationStructureBuildTypeKHR::DEVICE_KHR,
                &build_info,
                &max_primitive_counts,
            )
        };

        AccelerationStructureBuildSizesInfo {
            acceleration_structure_size: build_sizes.acceleration_structure_size,
            update_scratch_size: build_sizes.update_scratch_size,
            build_scratch_size: build_sizes.build_scratch_size,
        }
    }

    pub fn create_acceleration_structure(
        &self,
        info: AccelerationStructureInfo,
    ) -> AccelerationStructure {
        let acceleration_structure = unsafe {
            self.handle()
                .create_acceleration_structure_khr(
                    &vk::AccelerationStructureCreateInfoKHRBuilder::new()
                        ._type(info.level.to_erupt())
                        .offset(info.region.offset)
                        .size(info.region.size)
                        .buffer(info.region.buffer.handle()),
                    None,
                )
                .unwrap()
        };

        self.inner
            .acceleration_structures
            .lock()
            .insert(acceleration_structure);

        let device_address = DeviceAddress::new(unsafe {
            self.handle().get_acceleration_structure_device_address_khr(
                &vk::AccelerationStructureDeviceAddressInfoKHRBuilder::new()
                    .acceleration_structure(acceleration_structure),
            )
        });

        AccelerationStructure::new(info, acceleration_structure, device_address)
    }
}

fn get_allocator_memory_usage(usage: &vk::ImageUsageFlags) -> UsageFlags {
    if usage.contains(vk::ImageUsageFlags::TRANSIENT_ATTACHMENT) {
        UsageFlags::TRANSIENT
    } else {
        UsageFlags::empty()
    }
}

fn copy_group_handlers(
    group_handlers: &[u8],
    write: &mut [u8],
    group_indices: impl IntoIterator<Item = u32>,
    write_offset: &mut usize,
    group_size: u64,
    group_stride: usize,
) -> Option<Range<u64>> {
    let result_start = u64::try_from(*write_offset).ok()?;
    let group_size_usize = usize::try_from(group_size).ok()?;

    for group_index in group_indices {
        let group_offset = (group_size_usize.checked_mul(usize::try_from(group_index).ok()?))?;

        let group_end = group_offset.checked_add(group_size_usize)?;
        let write_end = write_offset.checked_add(group_size_usize)?;

        let group_range = group_offset..group_end;
        let write_range = *write_offset..write_end;

        let handler = &group_handlers[group_range];
        let output = &mut write[write_range];

        output.copy_from_slice(handler);
        *write_offset = write_offset.checked_add(group_stride)?;
    }

    let result_end = u64::try_from(*write_offset).ok()?;
    Some(result_start..result_end)
}
